# coding=utf-8

# yapf: disable
# noqa
# isort: skip_file

# Modifications Copyright 2024 by SambaNova Systems, Inc. All rights reserved.

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

# ---------------
# The patch file for modelzoo patch
# We modify some of the functions from the original code to better suit
# the SambaNova stack.
#
import math
from abc import ABC
from typing import Optional, Tuple, Union, Dict, List

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from sambanova_modelzoo.models.config import SNPretrainedConfig
from sambanova_modelzoo.models.custom_ops import addmm, create_3d_attn_mask, triu_fill, Tensor, upper_triangular_fill, sliding_window_fill
from sambanova_modelzoo.models.directives import add_directives
from sambanova_modelzoo.libs.nlp.core.directives import sdpa_directives
from sambanova_modelzoo.models.modeling_utils import (
    apply_rotary_pos_emb,
    get_position_ids,
    get_sliced_hidden_states,
    update_kv_cache,
    get_cos_sin_cache,
    TensorCache
)
from sambanova_modelzoo.models.modeling_patch_utils import MASK_MIN_VALUE, finfo_float32_min_patch
from sambanova_modelzoo.models.patch_router import (
    sn_patch_replace,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from sambanova_modelzoo.models.mistral.heuristics.hyperfunction_mistral import MistralHyperfunction

logger = logging.get_logger(__name__)
tensor_cache = TensorCache()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sn_mistral_rms_norm_forward(self, hidden_states):
    """
    Applies the Mistral RMS normalization layer to the input hidden states.

    Args:
        hidden_states (torch.Tensor): The input hidden states tensor.

    Returns:
        torch.Tensor: The normalized hidden states tensor.

    Notes:
        This function casts the input hidden states to fp32 optionally to enable RDU mixed precision computation.
        It then calculates the variance of the hidden states, and normalizes them using the calculated variance.
        The weight of the normalization layer is also applied to the normalized hidden states.

    """
    # [SambaNova] cast to fp32 optionally to enable RDU mixed precision computation
    hidden_states = hidden_states.to(torch.float32 if self.fp32_ln else torch.bfloat16)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states

def _sn_mistral_set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def sn_mistral_rotary_embedding_forward(self, x, seq_len=None):
    """
    Applies the Mistral rotary embedding layer to the input hidden states.

    Args:
        x (torch.Tensor): The input hidden states tensor with shape (batch, num_attention_heads, seq_len, head_size).
        seq_len (int, optional): The sequence length. If not provided, it will be inferred from the input tensor.

    Returns:
        tuple: A tuple containing the cosine and sine embeddings.

    Notes:
        This function calculates the cosine and sine embeddings using the cached values if the sequence length is within the cached range.
        If the sequence length is larger than the cached range, it recalculates the embeddings and updates the cache.

    """
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

    if seq_len < self.max_seq_len_cached:
        cos_slice = self.cos_cached.split(seq_len, dim=0)[0]
        sin_slice = self.sin_cached.split(seq_len, dim=0)[0]
    else:
        cos_slice = self.cos_cached
        sin_slice = self.sin_cached

    return (
        cos_slice.to(dtype=x.dtype),
        sin_slice.to(dtype=x.dtype),
    )

def sn_mistral_mlp_forward(self, x):
    with add_directives({'opfusion_id': 'up_proj'}): 
            up_proj = self.up_proj(x)

    with add_directives({'opfusion_id': 'gate_proj'}): 
        gate_proj = self.gate_proj(x)

    result = self.act_fn(gate_proj) * up_proj

    with add_directives({'opfusion_id': 'down_proj'}): 
        down_proj = self.down_proj(result)

    return down_proj


def sn_mistral_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        is_causal: bool,
        dropout_p: float,
        mixedp_attn: bool,
        seg_softmax_block_size: Optional[int],
        sliding_window_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ##SN: Description of SDPA can be found in go/flash-attention or go/air-sdpa-mapping
    Args:
        q: Query states, input to sdpa.
        k: Key states, input to sdpa.
        v: Value states, input to sdpa.
        attention_mask: Attention mask, input to sdpa.\
                        If is a tuple, it will be viewed as a pair of collapsed 3D article mask.
        is_causal: Whether to use causal masking, input to sdpa.
        dropout_p: Dropout percentage, input to the sdpa.
        mixedp_attn: Controls the precision of the computation:
                     False: pure bf16/fp32 (depending on input dtype)
                     True: conservative mixp, first matmul is bf16->fp32, softmax maxbias is fp32, softmax exp is bf16
    
    Returns:
        A tuple of tensors. The first tensor is the output of sdpa, second tensor is attention score, the result of softmax in sdpa.
        The second tensor is garbage when `use_segmented_softmax_attn` is True.
    
    Restrictions:
        `is_causal` and non-3D `attention_mask` are mutually exclusive
         Assumes 4D q, k, v
    """
    has_attn_mask = attention_mask is not None

    use_3d_attention = False
    # Do not support 3d attention with Mistral for now
    # Reconstruct normal mask from collapsed 3d article attention mask, adding causal mask if needed.
    if type(attention_mask) is tuple and len(attention_mask) == 2:
        attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)
        use_3d_attention = True

    if attention_mask is not None:
        # make sure attention_mask has shape (bs, n_heads, q_ss, kv_ss)
        bs, n_heads, q_ss = q.shape[:3]
        kv_ss = k.shape[2]
        assert attention_mask.shape[-1] == kv_ss, \
            f"Last dimension of attention_mask should be kv_seq_length. The mask has shape {attention_mask.shape} and expected last two dimensions is {kv_ss}."
        if len(attention_mask.shape) == 2:  # 2D cached attn mask: (mask_bs, kv_ss) -> (mask_bs, q_ss, kv_ss)
            attention_mask = torch.unsqueeze(attention_mask, 1)
        if len(attention_mask.shape) == 3:  # 3D attn mask: (mask_bs, q_ss, kv_ss) -> (mask_bs, n_heads, q_ss, kv_ss)
            attention_mask = torch.unsqueeze(attention_mask, 1)

        if mixedp_attn or (attention_mask.dtype != q.dtype and q.dtype == torch.float32):
            attention_mask = attention_mask.to(torch.float32)

        if use_3d_attention:
            attention_mask = sliding_window_fill(attention_mask, MASK_MIN_VALUE, sliding_window_size, mixedp_attn)
            is_causal = False
            sliding_window_size = -1

    if attention_mask is not None:
        if mixedp_attn :
            attention_mask = attention_mask.to(torch.float32)
        elif not (attention_mask.dtype == q.dtype and q.dtype == torch.float32):
            attention_mask = attention_mask.to(torch.bfloat16)
        is_causal = False

    assert not (is_causal and has_attn_mask), "Causal mask and Non-3D article attention mask are mutually exclusive."
    with sdpa_directives({
            'sdp_mixed_p': mixedp_attn,
            'sdp_block_size': seg_softmax_block_size,
            'sdp_sliding_window_size': (sliding_window_size, sliding_window_size)
    }):
        with add_directives({'opfusion_id': 'sdpa'}): 
            attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p, is_causal=is_causal)
        return attn_output, attn_output


@sn_patch_replace(
    patch=sn_mistral_sdpa,
    enable_if=lambda config: config.use_segmented_softmax_attn,
    description='Use SN fast implementation of SDPA.')
def sn_mistral_non_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        is_causal: bool,
        dropout_p: float,
        mixedp_attn: bool,
        sliding_window_size: int = 4096,
        seg_softmax_block_size=4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention weights and output using Mistral non-SDPA.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        attention_mask: Attention mask tensor. Can be a tuple of two tensors for 3D attention.
        is_causal: Whether to use causal attention.
        dropout_p: Dropout probability.
        mixedp_attn: Controls the precision of the computation.
        sliding_window_size: Sliding window size for 3D attention.
        seg_softmax_block_size: Segment softmax block size.

    Returns:
        A tuple of tensors. The first tensor is the output of sdpa, second tensor is attention score, the result of softmax in sdpa.
        The second tensor is garbage when `use_segmented_softmax_attn` is True.

    Restrictions:
        `is_causal` and non-3D `attention_mask` are mutually exclusive
         Assumes 4D q, k, v
    """
    bsz, num_heads, q_len, head_dim = q.shape
    kv_seq_len = v.shape[-2]

    trans = k.transpose(2, 3)
    with add_directives({'opfusion_id': 'matmul'}): 
        attn_weights = torch.matmul(q, trans)

    # mixedp_attn
    attn_weights = attn_weights.float()

    attn_weights = attn_weights / math.sqrt(head_dim)

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}")

    if isinstance(attention_mask, tuple):
        # 3D on-chip mask generation
        # apply the block diagonal 3D mask, the masked_fill operation above did the upper triangular masking
        # already, which allows us to use a block diagonal 3D mask instead of requiring us to construct a lower
        # triangular block diagonal mask
        attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)

    # using diagonal fill
    if is_causal:
        attn_weights = triu_fill(attn_weights, float('-inf'))

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")

        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    # mixedp_attn downcast before second matmul
    if mixedp_attn:
        attn_weights = attn_weights.bfloat16()
    with add_directives({'opfusion_id': 'matmul_1'}): 
        attn_output = torch.matmul(attn_weights, v)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                        f" {attn_output.size()}")

    return attn_output, attn_weights


def sn_mistral_attention_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            padding_mask: Optional[torch.Tensor] = None,
            last_token_index: Optional[torch.Tensor] = None,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for the Mistral attention mechanism.

        This function applies the attention mechanism to the input hidden states, using the query, key, and value projections.
        It also handles caching and padding for efficient inference and training.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position IDs. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): The cached key-value pairs. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
            use_cache (bool, optional): Whether to use caching. Defaults to False.
            padding_mask (Optional[torch.Tensor], optional): The padding mask. Defaults to None.
            last_token_index (Optional[torch.Tensor], optional): The last token index. Defaults to None.
            cos (Optional[torch.Tensor], optional): The cosine values for rotary position embedding. Defaults to None.
            sin (Optional[torch.Tensor], optional): The sine values for rotary position embedding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: The output attention weights, attention weights, and cached key-value pairs.
        """
        bsz, q_len, _ = hidden_states.size()
        with add_directives({'opfusion_id': 'q_proj'}): 
            query_states = self.q_proj(hidden_states)
        
        with add_directives({'opfusion_id': 'k_proj'}): 
            key_states = self.k_proj(hidden_states)
        
        with add_directives({'opfusion_id': 'v_proj'}): 
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        with add_directives({'opfusion_id': 'value_states'}):
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value_provided = past_key_value is not None

        consume_cache = use_cache and past_key_value_provided

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value_provided:
            # [SambaNova] cached inference token-by-token generation after prompt
            layer_past_key, layer_past_value = past_key_value

            if self.config.mixedp_attn:
                layer_past_key = layer_past_key.bfloat16()
                layer_past_value = layer_past_value.bfloat16()
                key_states = key_states.bfloat16()
                value_states = value_states.bfloat16()
            
            key_states, value_states = update_kv_cache(layer_past_key, layer_past_value, key_states, value_states,
                                                        last_token_index)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # [SambaNova] cache gen on the prompt / non-cached inference / training
            past_key_value = None
            if use_cache:
                key_states_padded, value_states_padded = key_states, value_states
                if self.config.max_seq_length is not None and q_len < self.config.max_seq_length:
                    shape = (bsz, self.num_key_value_heads, self.config.max_seq_length - q_len, self.head_dim,)
                    zero_padding = tensor_cache.get_or_create_zeroes(key=f'zero_padding_{shape}', shape=shape,
                                                                     dtype=key_states.dtype)
                    key_states_padded = torch.cat((key_states, zero_padding), dim=2)
                    value_states_padded = torch.cat((value_states, zero_padding), dim=2)
                past_key_value = (key_states_padded, value_states_padded)

        # repeat k/v heads if n_kv_heads < n_heads
        with add_directives({'opfusion_id': 'reshape_kv'}): 
            key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        is_causal = not consume_cache

        attn_output, attn_weights = sn_mistral_non_sdpa(
            self,
            q=query_states,
            k=key_states,
            v=value_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=self.config.attention_dropout if self.training else 0,
            mixedp_attn=self.config.mixedp_attn,
            sliding_window_size=self.config.sliding_window if not past_key_value_provided else -1,
            seg_softmax_block_size=self.config.seg_softmax_block_size)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        with add_directives({'opfusion_id': 'o_proj'}): 
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def sn_mistral_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            padding_mask: Optional[torch.Tensor] = None,
            last_token_index: Optional[torch.Tensor] = None,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for the Mistral decoder.

        This function applies the Mistral decoder to the input hidden states, using the provided attention mask, position IDs, past key-value pairs, and other optional inputs.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position IDs. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): The past key-value pairs. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to False.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
            padding_mask (Optional[torch.Tensor], optional): The padding mask. Defaults to None.
            last_token_index (Optional[torch.Tensor], optional): The last token index. Defaults to None.
            cos (Optional[torch.Tensor], optional): The cosine values. Defaults to None.
            sin (Optional[torch.Tensor], optional): The sine values. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: A tuple containing the output hidden states and optional attention weights and past key-value pairs.
        """
        residual = hidden_states
        # fp32_skip_add
        if self.config.fp32_skip_add:
            residual = residual.float()

        with add_directives({'opfusion_id': 'input_layernorm'}): 
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states,
                                                                             attention_mask=attention_mask,
                                                                             position_ids=position_ids,
                                                                             past_key_value=past_key_value,
                                                                             output_attentions=output_attentions,
                                                                             use_cache=use_cache,
                                                                             padding_mask=padding_mask,
                                                                             last_token_index=last_token_index,
                                                                             cos=cos,
                                                                             sin=sin)
        with add_directives({'opfusion_id': 'o_proj_add'}):
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # fp32_skip_add
        if self.config.fp32_skip_add:
            residual = residual.float()
        with add_directives({'opfusion_id': 'post_attention_layernorm'}):
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        with add_directives({'opfusion_id': 'mlp_add'}):
            hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


def sn_mistral_model_self(sn_mistral_model: 'SNMistralModel', rotary_embedding: 'MistralRotaryEmbedding'):
    # [SambaNova] Set hyperfunction boundaries
    sn_mistral_model.hyperfunction = MistralHyperfunction(sn_mistral_model.config)
    # [SambaNova] Calculate cos/sin only once.
    sn_mistral_model.head_dim = sn_mistral_model.config.hidden_size // sn_mistral_model.config.num_attention_heads
    sn_mistral_model.max_position_embeddings = sn_mistral_model.config.max_position_embeddings
    sn_mistral_model.rope_theta = sn_mistral_model.config.rope_theta
    sn_mistral_model.rotary_emb = rotary_embedding(
        sn_mistral_model.head_dim,
        max_position_embeddings=sn_mistral_model.max_position_embeddings,
        base=sn_mistral_model.rope_theta,
    )


def sn_mistral_model_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            last_token_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for the Mistral decoder.

        This function applies the Mistral decoder to the input hidden states, using the provided attention mask, position IDs, past key-value pairs, and other optional inputs.

        Args:
            input_ids (torch.Tensor): The input IDs.
            inputs_embeds (Optional[torch.Tensor], optional): The input embeddings. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position IDs. Defaults to None.
            past_key_values (Optional[Tuple[torch.Tensor]], optional): The past key-value pairs. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to False.
            output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to False.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
            padding_mask (Optional[torch.Tensor], optional): The padding mask. Defaults to None.
            last_token_index (Optional[torch.Tensor], optional): The last token index. Defaults to None.
            cos (Optional[torch.Tensor], optional): The cosine values. Defaults to None.
            sin (Optional[torch.Tensor], optional): The sine values. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a BaseModelOutputWithPast object. Defaults to True.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: A tuple containing the output hidden states, next cache, all hidden states, all self-attention weights, and last hidden states before norm, or a BaseModelOutputWithPast object if return_dict is True.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # [SambaNova]: defines a scope for optimized mapping. No effect if not using heuristic mapping
        consume_cache = use_cache and past_key_values is not None

        with self.hyperfunction.embedding(seq_length, consume_cache, self.training):
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                # [SambaNova] generative inference does not need to pass position_ids because we use right padding
                position_ids = get_position_ids(batch_size, seq_length, last_token_index, device)
            position_ids = position_ids.view(-1, seq_length).long()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if not getattr(self.config, "use_segmented_softmax_attn", False) and attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length),
                                            dtype=torch.bool,
                                            device=inputs_embeds.device)
            padding_mask = None

            if (consume_cache or not getattr(self.config, "use_segmented_softmax_attn", False)) and not isinstance(attention_mask, tuple):
                # 4d mask is passed through the layers
                with finfo_float32_min_patch():
                    attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length),
                                                                       inputs_embeds, past_key_values_length)

            # [SambaNova] This makes sure the embedding between each layer has the same dtype
            if self.config.fp32_skip_add and self.config.use_plugin_heuristics:
                inputs_embeds = inputs_embeds.float()

        hidden_states = inputs_embeds

        with self.hyperfunction.embedding(seq_length, consume_cache, self.training, reuse_last_id=True):
            seq_len = max(seq_length, past_key_values_length)
            cos, sin = tensor_cache.get_or_create_tuple(
                f"{seq_len}_cos_sin_tensors", 
                lambda: self.rotary_emb(hidden_states, seq_len=seq_len)
            )
            cos, sin = get_cos_sin_cache(cos, sin, position_ids)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            with self.hyperfunction.encoder(seq_length, consume_cache):
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                    )
                else:
                    layer_outputs = decoder_layer(hidden_states,
                                                  attention_mask=attention_mask,
                                                  position_ids=position_ids,
                                                  past_key_value=past_key_value,
                                                  output_attentions=output_attentions,
                                                  use_cache=use_cache,
                                                  padding_mask=padding_mask,
                                                  last_token_index=last_token_index,
                                                  cos=cos,
                                                  sin=sin)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        with self.hyperfunction.classifier(seq_length, consume_cache, self.training):
            # [SambaNova] Gather cannot be tiled along the gathering dimension
            # having this at the beginning of the classifier section allow the rest
            # of the classifier be pipelined without materalizing the full tensor
            if last_token_index is not None and hidden_states.shape[1] > 1:
                hidden_states = get_sliced_hidden_states(hidden_states, last_token_index)
            # [SambaNova]: Reutrn hidden_states before norm for performant mapping through fusing norm, lm_head and sampling.
            last_hidden_states_before_norm = hidden_states
            with add_directives({'opfusion_id': 'norm'}): 
                hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, last_hidden_states_before_norm] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def sn_mistral_for_causallm_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            last_token_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the causal language model.

        This method performs a forward pass for the causal language model, taking into account the input IDs, attention mask, position IDs, past key values, and other optional inputs.

        Args:
            input_ids (torch.LongTensor): The input IDs for the model.
            attention_mask (Optional[torch.Tensor]): The attention mask for the model.
            position_ids (Optional[torch.LongTensor]): The position IDs for the model.
            past_key_values (Optional[List[torch.FloatTensor]]): The past key values for the model.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings for the model.
            labels (Optional[torch.LongTensor]): The labels for the model.
            use_cache (Optional[bool]): Whether to use the cache for the model.
            output_attentions (Optional[bool]): Whether to output the attentions for the model.
            output_hidden_states (Optional[bool]): Whether to output the hidden states for the model.
            return_dict (Optional[bool]): Whether to return a dictionary for the model.
            last_token_index (Optional[torch.Tensor]): The last token index for the model.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: The output of the model, which can be a tuple of logits, loss, and other outputs, or a CausalLMOutputWithPast object.

        Raises:
            ValueError: If the input IDs are not of the correct shape.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape
        consume_cache = use_cache and past_key_values is not None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            last_token_index=last_token_index,
        )

        hidden_states = outputs[0]

        with self.hyperfunction.classifier(seq_length, consume_cache, self.training, reuse_last_id=True):
            with add_directives({'opfusion_id': 'lm_head'}): 
                logits = self.lm_head(hidden_states)
            if self.config.fp32_logits:
                logits = logits.float()

            loss = None
            if labels is not None:
                # [SambaNova] slicing is not efficient on RDU, so we expect the logits and labels are already shifted during
                # dataloader preprocessing. No need to shift again here.
                shift_logits = logits.view(-1, self.config.vocab_size)
                shift_labels = labels.view(-1)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits, ) + outputs[1:]
                return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
