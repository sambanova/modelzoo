# Copyright 2024 SambaNova Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Tuple

import torch

from .custom_ops import gather, scatter


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims (last dim) of the input.
    Args:
        x: Rotary embedded tensor
    Return:
        Tensor with half of last dim negated and rotated to the front.
    """
    # [SambaNova] indexselect support is limited and not as performant as split
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def get_cos_sin_cache(cos: torch.Tensor, sin: torch.Tensor,
                      position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the original cos/sin cache, return the positioned cos/sin cache to be applied with query a key.
    This part of code was originally with apply_rotary_pos_emb. It's taken out separately because the slicing on
    position_ids is not dependent on each encoder layer. The memory view changes / expand/ slicing is not free on RDU.
    Doing it in every encoder layer is costly. Thus this function is extracted out and to be called just once at the
    beginning of the model.

    Args:
        cos: Cos cache with max_position_embedding length. Of size (max_seq_len, head_dim).
        sin: Sin cache with max_position_embedding length. Same size as cos.
        position_ids: Position ids of the inputs. Of size(batch_size or 1, input_seq_len).
    Returns:
        Positioned cos and sin cache.
    """
    # [SambaNova] Samba index select support is limited
    # todo: expand this and move support into compiler
    # the first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    # sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    assert len(cos.shape) == 2, f"cos should be of shape (max_seq_len, head_dim), got {cos.shape} instead"
    assert sin.shape == cos.shape, f"sin should be of shape (max_seq_len, head_dim), got {sin.shape} instead"
    assert len(position_ids.shape) == 2, f"position_ids should be of 2D, got {position_ids.shape} instead"

    # [SambaNova] index select support uses gather, which is often not as efficient as using gather directly
    # cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    # sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    # [SambaNova] Add one more dim that to be used by gather
    position_ids = position_ids.unsqueeze(-1)
    cos = gather(cos, position_ids, [0], [1])
    sin = gather(sin, position_ids, [0], [1])
    cos = cos.permute(0, 2, 1, 3)
    sin = sin.permute(0, 2, 1, 3)
    return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding (cos, sin) to the query and key tensor on the sequence dimension.

    The legends for dimensions are defined as:
    num_heads: number of attention heads
    current_seq_len: the current batch's sequence length, should be either 1 or max_seq_len
    max_seq_len: the static sequence length, different from current_seq_len in cached inference case where it is always
                 maximum lenghth, e.g. the length of static sequence length of KV cache

                 
    Args:
        q: Query tensor, of size (batch_size, num_heads, current_seq_len, head_dim)
        k: Key tensor, of size (batch_size, num_key_value_heads, current_seq_len, head_dim)
        cos: Cosine base of rotary embedding, of size (max_seq_len, head_dim)
        sin: Sine base of rotary embedding, of size (max_seq_len, head_dim)
        position_ids: The position indices of the tokens corresponding to the query and key tensors. It has a size of
                      (batch_size, current_seq_len).

    Returns:
        Embedded query and key tensor of same size as input.
    
    """
    bs, nheads, cur_seq_len, head_dim = q.shape
    assert len(
        k.shape) == 4, f"k should be of shape (batch_size, num_heads, current_seq_len, head_dim), got {k.shape} instead"
    assert k.shape[0] == bs, f"k has a different batch_size {k.shape[0]} compared to q {bs}"
    assert list(k.shape[2:]) == [cur_seq_len,
                                 head_dim], f"k has different current_seq_len and/or head_dim compared to q"
    assert cos.shape[3] == head_dim, f"cos should have dim of head dim {head_dim}, got {cos.shape[3]} instead"
    assert list(position_ids.shape) in [[bs, cur_seq_len], [1, cur_seq_len]],\
            f"position_ids should be of shape {[bs, cur_seq_len]} or {[1, cur_seq_len]}, got {position_ids.shape} instead"

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_position_ids(batch_size: int, input_seq_length: int, last_token_index: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    For better performance, SambaNova's stack requires a static graph which in turn requires a fixed sequence length.
    So padding is always needed to satisfy the fixed sequence length with only one exception: sequence length of 1.
    Sequence length of 1 is used in the continuous token generation of cached inference.

    For simplicity and performance, we decided to only support right padding not left padding on the inputs. That means 
    we can always generate a fixed position_ids that can be used by any input except sequence length of 1. For input
    sequence lenght of 1, e.g. continuous token generation,  the last_token_index has the same meaning as input's position_ids.

    Args:
        batch_size: Batch size
        input_seq_length: Sequence length of input, either static max seq length or 1
        last_token_index: The actual position of the last token. For cached inference's continous token generation,
        it corresponds to the real position of the input token considering all previous generated tokens and prompt. 
        It has a shape of (batch_size,). 

    Returns:
        position_ids of shape (batch_size, input_seq_length)
    """
    if last_token_index is not None:
        assert list(last_token_index.shape) == [batch_size, 1], 'last_token_index must be of shape (batch_size, 1)'

    if input_seq_length == 1:
        position_ids = last_token_index
    else:
        # [SambaNova] Since only right padding is supported,  we can always prepopulate position_ids
        position_ids = torch.arange(0, input_seq_length, device=device, dtype=int).reshape(1, -1).expand(batch_size, -1)
    return position_ids


def get_sliced_hidden_states(hidden_states: torch.Tensor, last_token_index: torch.Tensor) -> torch.Tensor:
    """
    Slice the hidden states on last_token_index for first token generation of cached inference.
    This is to avoid transferring large logits from device to host memory. And only the
    last token index is needed to generate the new token.

    Args:
        hidden_states: Hidden states of size (batch_size, seq_len, dim)
        last_token_index: The actual position of the last token. For cached inference's continous token generation,
        it corresponds to the real position of the input token considering all previous generated tokens and prompt. 
        It has a shape of (batch_size,). 

    Returns:
        Sliced hidden states of size (batch_size, 1, dim)
    """
    assert len(hidden_states.shape) == 3,\
            f"hidden states 3D of shape [batch_size, seq_len, dim] but got {hidden_states.shape}"
    assert list(last_token_index.shape) == [hidden_states.shape[0], 1],\
            f"last_token_index must be of shape {[hidden_states.shape[0], 1]} but got {last_token_index.shape}"
    last_token_index = last_token_index.unsqueeze(1)
    # [SambaNova] using gather is often more efficient than doing slicing
    hidden_states = gather(hidden_states, last_token_index, [0], [1], batched_dims=1)
    # [SambaNova] gather will introduce a dummy dimension on the gathering dim, squeeze it out
    hidden_states = hidden_states.squeeze(-2)
    return hidden_states


def update_kv_cache(key_cache: torch.Tensor, value_cache: torch.Tensor, current_key: torch.Tensor,
                    current_value: torch.Tensor, last_token_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update individual decoder layer's key_cache and value_cache with current_key and current_value at last_token_index.
    The key value caches are of static sequence length.
    positions.
    Args:
        key_cache: Past key cache, of shape (batch_size, num_heads, max_seq_length, head_dim) 
        value_cache: Past value cache, of shape (batch_size, num_heads, max_seq_length, head_dim) 
        current_key: Current key states, of shape (batch_size, num_heads, 1, head_dim)
        current_value: Current key states, of shape (batch_size, num_heads, 1, head_dim)
        last_token_index: The position to update the current key/value, of shape (batch_size,)
    Returns:
        Updated key and value cache.
    
    """
    assert len(key_cache.shape) == 4, 'key_cache must be of shape (batch_size, num_heads, max_seq_length, head_dim)'
    assert value_cache.shape == key_cache.shape, 'value_cache must be of same shape as key_cache'
    cache_shape = key_cache.shape
    assert list(current_key.shape) == [cache_shape[0], cache_shape[1], 1, cache_shape[3]],\
            'current_key must be of shape (batch_size, num_heads, 1, head_dim)'
    assert current_value.shape == current_key.shape, 'current_value must be same shape as current key'
    assert list(last_token_index.shape) == [cache_shape[0], 1], 'last_token_index must be of shape (batch_size, 1)'

    last_token_index = last_token_index.unsqueeze(1)
    current_key = current_key.unsqueeze(1)
    current_value = current_value.unsqueeze(1)

    key_cache = scatter(key_cache, current_key, start_indices=last_token_index, scatter_dims=[1], batched_dims=1)
    value_cache = scatter(value_cache, current_value, start_indices=last_token_index, scatter_dims=[1], batched_dims=1)

    return key_cache, value_cache


class GlobalNamedTensor:
    _global_tensors: Dict[str, torch.Tensor] = {}

    @classmethod
    def convert(cls, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """
        First call of this on a a specific name, will return the input itself as well as storing as it as global instance.
        The following call with the same name will return the stored global tensor instance to overwrite the input.
        Exception will be raised if the input tensor mismatches the global one. This is useful for sharing the same 
        device memory when compiling onto RDU.

        Args:
            tensor: Tensor to be stored as global instance (first call) or verified against global instance 
                    (after first call).
            name: Global name id.
        Returns:
            Global input tensor instance.
        """
        if name in cls._global_tensors:
            if not torch.equal(cls._global_tensors[name], tensor):
                raise ValueError(
                    f"Input named tensor {name}:\n{tensor}\ndoes not match the cached one:\n{cls._global_tensors[name]}"
                )
            return cls._global_tensors[name]
        cls._global_tensors[name] = tensor
        return tensor
