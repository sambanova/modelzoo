# Modifications Copyright 2024 by SambaNova Systems, Inc. All rights reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Gemma model."""
import math
from typing import List, Optional, Tuple, Union
from abc import ABC

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from sambanova_modelzoo.models.custom_ops import create_3d_attn_mask, triu_fill, upper_triangular_fill
from sambanova_modelzoo.models.gemma.heuristics.hyperfunction_gemma import GemmaHyperfunction
from sambanova_modelzoo.models.gemma.configuration_gemma import SNGemmaConfig
from sambanova_modelzoo.libs.nlp.core.directives import sdpa_directives
from sambanova_modelzoo.models.modeling_patch_utils import MASK_MIN_VALUE, finfo_float32_min_patch
from sambanova_modelzoo.models.modeling_utils import (apply_rotary_pos_emb, get_cos_sin_cache, get_position_ids,
                                     get_sliced_hidden_states, update_kv_cache, TensorCache)
from sambanova_modelzoo.models.utils import logger_info
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)
tensor_cache = TensorCache()

"""
This Gemma patch is based off Huggingface with transformers==4.38.1

The following are specific to SambaNova, not from original model code.
To use the model:
    1. use_diagonal_mask, used on causual tasks, i.e. attention is restricted to the past tokens, set is_diagonal_mask to use on-chip
    mask generation so you don't need to pass the mask explicitly. It can be used in both training and inference. For inference,
    diagonal masking is used to mask out the padding tokens.
    2. 3D attention is used to restrict attention to only attention the current article/example, see create_3d_attn_mask, pass the attention_mask as Tuple[Tensor, Tensor] to enable this. 3D attention has to be used with diagonal_mask but diagonal_mask can be used alone to do causal attention.
    3. Prepare inputs and labels to be already shifted so that tokens < n predict n. Because shifting/slicing is not very performant on RDU yet. We recommend you either shift both inputs and labels similar to the original huggingface model:
        sentence = sentence[..., :-1]
        labels = labels[..., 1:]
       Or just shift the labels and use CorssEntropyLoss to ignore the last token prediction, for example,
        labels = labels[..., 1:]
        labels = torch.cat((labels, torch.ones([labels.shape[0], 1], dtype=labels.dtype) * -100), dim=1)
       where -100 is the ignore_index for CrossEntropyLoss.

                                            use_cache        use_diagonal_mask
cached_inference_prompt
cached_inference_token_generation
none_cached_inference
training

TODO: expand this
"""


def gemma_mha(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              attention_mask: Optional[torch.Tensor],
              is_causal: bool,
              dropout_p: float,
              mixedp_attn: bool,
              use_gqa: bool = False):
    """
    Huggingface version of normal MHA, e.g. GemmaAttention
    NOTE: Key should be already transposed for the first matmul.
    """
    if use_gqa:
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim = q.shape
        num_heads = num_kv_heads * num_kv_groups
    else:
        bsz, num_heads, q_len, head_dim = q.shape
    kv_seq_len = v.shape[-2]

    attn_weights = torch.matmul(q, k)
    # [SambaNova] we need to do float until softmax, softmax is in mixed precision
    if mixedp_attn:
        attn_weights = attn_weights.to(torch.float)
    attn_weights = attn_weights / math.sqrt(head_dim)

    if use_gqa:
        if attn_weights.size() != (bsz, num_kv_heads, num_kv_groups, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_kv_heads, num_kv_groups, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")
    else:
        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                             f" {attn_weights.size()}")

    if isinstance(attention_mask, tuple):
        # 3D on-chip mask generation
        # apply the block diagonal 3D mask, the masked_fill operation above did the upper triangular masking
        # already, which allows us to use a block diagonal 3D mask instead of requiring us to construct a lower
        # triangular block diagonal mask
        attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)

    if is_causal:
        attn_weights = triu_fill(attn_weights, float('-inf'))

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
        if use_gqa:
            attention_mask = attention_mask.unsqueeze(dim=1)

        attn_weights = attn_weights + attention_mask

    # [SambaNova] use mixedp softmax not float32
    # upcast attention to fp32
    # TODO: understand how compiler lowers dtype=float32 ops in O3 mode
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    if not use_gqa:
        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                             f" {attn_output.size()}")
    else:
        if attn_output.size() != (bsz, num_kv_heads, num_kv_groups, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_kv_heads, num_kv_groups, q_len, head_dim)}, but is"
                f" {attn_output.size()}")
    return attn_output, attn_weights


def gemma_sdpa_switch(q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
                      is_causal: bool,
                      dropout_p: float,
                      use_segmented_softmax_attn: bool,
                      seg_softmax_block_size: Optional[int],
                      mixedp_attn: bool,
                      key_transposed: Optional[bool] = None,
                      use_gqa: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Switches between different versions of scale dot product attention (SDPA)
    ##SN: Description of SDPA can be found in go/flash-attention or go/air-sdpa-mapping
    Args:
        q: Query states, input to sdpa.
        k: Key states, input to sdap.
        v: Value states, input to sdpa.
        attention_mask: If is a tuple, it will be viewed as a pair of collapsed 3D article mask, input to sdpa.
        is_causal: Whether to use causal masking, input to sdpa.
        dropout_p: Dropout percentage, input to the sdpa.
        use_segmented_softmax_attn: Controls whether to use segmented softmax attention.
        seg_softmax_block_size: Controls the block size used in segmented softmax attention
        mixedp_attn: Controls the precision of the computation:
                     False: pure bf16/fp32 (depending on input dtype)
                     True: conservative mixp, first matmul is bf16->fp32, softmax maxbias is fp32, softmax exp is bf16
        key_transposed(Deprecated): If the key is already transposed for the first matmul.
        use_gqa: Whether to use group query attention

    Returns:
        A tuple of tensors. The first tensor is the output of sdpa, second tensor is attention score, the result of softmax in sdpa.
        The second tensor is garbage when `use_segmented_softmax_attn` is True.

    Restrictions:
        `is_causal` and non-3D `attention_mask` are mutually exclusive
         Assumes 4D q, k, v
    """
    has_attn_mask = attention_mask is not None
    if use_segmented_softmax_attn:
        # Reconstruct normal mask from collapsed 3d article attention mask, adding causal mask if needed.
        if key_transposed:  # air.SDPA transposes key inside
            k = k.transpose(2, 3)
        if type(attention_mask) is tuple and len(attention_mask) == 2:
            logger_info('Using 3D article attention mask in Gemma2 Model.')
            attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)

        if attention_mask is not None:
            # make sure attention_mask has shape (bs, n_heads, q_ss, kv_ss)
            bs, n_heads, q_ss = q.shape[:3]
            kv_ss = k.shape[2]
            assert attention_mask.shape[-1] == kv_ss and attention_mask.shape[-2] == q_ss, \
                f"Last two dimension of attention_mask should be q_seq_length and kv_seq_length. The mask has shape {attention_mask.shape} and expected last two dimensions is {(q_ss, kv_ss)}."
            if len(attention_mask.shape) == 3:  # (mask_bs, q_ss, kv_ss)
                attention_mask = torch.unsqueeze(attention_mask, 1)

            if mixedp_attn:
                attention_mask = attention_mask.to(torch.float32)

            if is_causal:
                # TODO: remove as right padding no longer needs attention_mask most of time
                attention_mask = upper_triangular_fill(attention_mask, MASK_MIN_VALUE, mixedp_attn)
                is_causal = False
                
        assert not (is_causal and has_attn_mask), "is_causal and article attention mask are mutually exclusive in sdpa"

        with sdpa_directives({
                'sdp_mixed_p': mixedp_attn,
                'sdp_block_size': seg_softmax_block_size,
        }):
            logger_info('Using air SDPA in Gemma2 Model.')
            o = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p, is_causal=is_causal)
            return o, o
    else:
        if key_transposed is not None:
            if not key_transposed:
                k = k.transpose(-1, -2)
        else:
            k = k.transpose(-1, -2)
        logger_info('Using patched regular attn in Gemma2 Model.')
        return gemma_mha(q, k, v, attention_mask, is_causal, dropout_p, mixedp_attn, use_gqa=use_gqa)

class GemmaRMSNormPatchNamespace(ABC):
    @staticmethod
    def patch_fp32_ln(self: 'GemmaRMSNorm', fp32_ln: bool):
        """
        Record the layernorm mixed precision setting
        """
        self.fp32_ln = fp32_ln

    @staticmethod
    def patch_forward(self: 'GemmaRMSNorm', hidden_states: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        # [SambaNova] instead of forcing to float32, we use the auto mixed precision
        # hidden_states = hidden_states.to(torch.float32)
        if hasattr(self, 'config'):
            # in-memory patch overwrites config only, does not have fp32_ln in self
            fp32_ln = self.config.fp32_ln
        else:
            fp32_ln = self.fp32_ln
        hidden_states = hidden_states.to(torch.float32 if fp32_ln else torch.bfloat16)

        output = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * (1 + self.weight)

class GemmaRotaryEmbeddingPatchNamespace(ABC):
    @staticmethod
    def patch_inv_freq(_self: "GemmaRotaryEmbedding"):
        inv_freq = 1.0 / (_self.base ** (torch.arange(0, _self.dim, 2, dtype=torch.int64).float() / _self.dim))
        _self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @staticmethod
    def patch_cos_sin_cache(_self: "GemmaRotaryEmbedding", seq_len, device, dtype):
        """[SambaNova] cache the sin and cos tensors."""
        _self.max_seq_len_cached = seq_len
        t = torch.arange(_self.max_seq_len_cached, device=device, dtype=_self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, _self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        _self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        _self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @staticmethod
    def patch_forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            GemmaRotaryEmbeddingPatchNamespace.patch_cos_sin_cache(self, seq_len=seq_len, device=x.device, dtype=x.dtype)

        # [SambaNova] slicing is not performant on RDU, use split instead, see below
        # [TODO] can we have more efficient slicing on RDU ? Because we moved slicing to CPU data preparation, why is slicing is even slower compared to CPU ?
        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
        if seq_len < self.max_seq_len_cached:
            cos_slice, *_ = self.cos_cached.split(seq_len, dim=0)
            sin_slice, *_ = self.sin_cached.split(seq_len, dim=0)
        else:
            cos_slice = self.cos_cached
            sin_slice = self.sin_cached

        return (
            cos_slice.to(dtype=x.dtype),
            sin_slice.to(dtype=x.dtype),
        )


class GemmaMLPPatchNamespace(ABC):
    @staticmethod
    def patch_hidden_act(_self: "GemmaMLP"):
        from sambanova_modelzoo.models.gemma.modeling_gemma import logger
        _self.act_fn = ACT2FN["gelu"]
        logger.warning_once(
            "JIT does not support the approximate GeLU activation function `gelu_pytorch_tanh`, automatically setting "
            "activation function to `gelu`."
        )


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


class GemmaAttentionPatchNamespace(ABC):
    @staticmethod
    def patch_mixp(self: "GemmaAttention"):
        self.mixedp_attn = self.config.mixedp_attn

    @staticmethod
    def patch_remove_rotary_emb(self: "GemmaAttention"):
        del self.rotary_emb

    @staticmethod
    def patch_forward(
        self: "GemmaAttention",
        hidden_states: torch.Tensor,
        # [SambaNova] attention mask can be a tuple to represent 3D on-chip mask gen
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # [SambaNova] Integer index tensor of size 1, indicates current token generation position for cached inference
        last_token_index: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value_provided = past_key_value is not None
        consume_cache = use_cache and past_key_value_provided

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value_provided:
            # [SambaNova] cached inference token-by-token generation after prompt
            layer_past_key, layer_past_value = past_key_value
            key_states, value_states = update_kv_cache(layer_past_key, layer_past_value, key_states, value_states,
                                                    last_token_index)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # [SambaNova] cache gen on the prompt / non-cached inference
            past_key_value = None
            if use_cache:
                key_states_padded, value_states_padded = key_states, value_states
                if self.config.max_seq_length is not None and q_len < self.config.max_seq_length:
                    shape = (bsz, self.num_key_value_heads, self.config.max_seq_length - q_len, self.head_dim)
                    zero_padding = tensor_cache.get_or_create_zeroes(key=f'zero_padding_{shape}', shape=shape,
                                                                     dtype=key_states.dtype)
                    key_states_padded = torch.cat((key_states, zero_padding), dim=2)
                    value_states_padded = torch.cat((value_states, zero_padding), dim=2)
                past_key_value = (key_states_padded, value_states_padded)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output, attn_weights = gemma_sdpa_switch(
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=not consume_cache,
            dropout_p=0,
            use_segmented_softmax_attn=self.config.use_segmented_softmax_attn,
            seg_softmax_block_size=self.config.seg_softmax_block_size,
            mixedp_attn=self.config.mixedp_attn,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class GemmaDecoderPatchNamespace(ABC):
    def patch_config(self: "GemmaDecoderLayer", config: SNGemmaConfig):
        self.config = config

    def patch_fp32_ln(self: "GemmaDecoderLayer", config: SNGemmaConfig):
        if not config.fp32_ln:
            raise ValueError('Incorrect config for fp32_ln')

        self.input_layernorm.fp32_ln = True
        self.post_attention_layernorm.fp32_ln = True

    def patch_forward(
            self: "GemmaDecoderLayer",
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            # [SambaNova] Integer index tensor of size 1, indicates current token generation position for cached inference
            last_token_index: Optional[torch.Tensor] = None,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        # [SambaNova] used to enable fp32 residual accumulation
        if self.config.fp32_skip_add:
            residual = residual.float()

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states,
                                                                            attention_mask=attention_mask,
                                                                            position_ids=position_ids,
                                                                            past_key_value=past_key_value,
                                                                            output_attentions=output_attentions,
                                                                            use_cache=use_cache,
                                                                            last_token_index=last_token_index,
                                                                            cos=cos,
                                                                            sin=sin)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # [SambaNova] used to enable fp32 residual accumulation
        if self.config.fp32_skip_add:
            residual = residual.float()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )
        return outputs

class SNGemmaModelPatchNamespace(ABC):
    def patch_rotary_emb(_self: "SNGemmaModel", config: SNGemmaConfig):
        """Initialize the rotary embedding for cached sin/cos"""
        from sambanova_modelzoo.models.gemma.modeling_gemma import GemmaRotaryEmbedding
        _self.rotary_emb = GemmaRotaryEmbedding(config.head_dim,
                                               max_position_embeddings=config.max_position_embeddings,
                                               base=config.rope_theta)

    def patch_fp32_ln(_self: "SNGemmaModel", config: SNGemmaConfig):
        if not config.fp32_ln:
            raise ValueError('Incorrect config for fp32_ln')
        _self.norm.fp32_ln = True

    def patch_hyperfunction(_self: "SNGemmaModel", config: SNGemmaConfig):
        """For O1 heuristic annotation"""
        _self.hyperfunction = GemmaHyperfunction(config)

    def patch_forward(
            self: "SNGemmaModel",
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # [SambaNova] Integer index tensor of size 1, indicates current token generation position for cached inference
            last_token_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # [SambaNova] Sanity check the attention mask.
        if self.training:
            err_msg = 'Attention Mask must be 3D or None (on-chip mask gen) during training'
            assert attention_mask is None or isinstance(attention_mask, tuple), err_msg

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        consume_cache = use_cache and past_key_values is not None

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
            assert past_key_values_length == self.config.max_seq_length, \
                    "max_seq_length must equal the length of past_key_value input in token_gen graph"

        with self.hyperfunction.embedding(seq_length, consume_cache, self.training):
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                # [SambaNova] generative inference does not need to pass position_ids because we use right padding
                position_ids = get_position_ids(batch_size, seq_length, last_token_index, device)

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            # FIXME: why the condition ?
            # [SambaNova] Only populate mask in either cached inference token-by-token generation or
            if consume_cache:
                # embed positions
                if attention_mask is None:
                    assert past_key_values is None
                    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)
                if not isinstance(attention_mask, tuple):
                    with finfo_float32_min_patch():
                        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length),
                                                                        inputs_embeds, past_key_values_length)

            hidden_states = inputs_embeds

            # normalized
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

            cos, sin = self.rotary_emb(hidden_states, position_ids, seq_len=max(seq_length, past_key_values_length))
            cos, sin = get_cos_sin_cache(cos, sin, position_ids)


            # [SambaNova] In terms of compute graph, the float cast on hidden_states is redundant to the float cast within
            # the first decoder forward call where the input hidden_states is cast to float again. This redundant cast is
            # to ensure O1 plugin heuristics's hyperfunction works for the first decoder.
            # Hyperfunction needs the inputs's metadata (includeing dtype) to be the same across layers to work. Without this
            # cast, the first decoder layer's hidden state input may look like bfloat16 while the rest decoder will always
            # have float. This will cause we compile a different hyperfunction for the first decoder layer which doubles the
            # compilation time and also the doubles the amount of work in heuristics design.
            if self.config.fp32_skip_add:
                hidden_states = hidden_states.float()

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

            with self.hyperfunction.encoder_decoder(seq_length, consume_cache):
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, use_cache=None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        last_token_index=last_token_index,
                        cos=cos,
                        sin=sin,
                    )
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
            # [SambaNova]: Return hidden_states before norm for performant mapping through fusing norm, lm_head and sampling.
            last_hidden_states_before_norm = hidden_states
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

class SNGemmaForCausalLMPatchNamespace(ABC):
    def patch_hyperfunction(_self: "SNGemmaForCausalLM", config: SNGemmaConfig):
        """For O1 heuristic annotation"""
        _self.hyperfunction = GemmaHyperfunction(config)
        _self.model.hyperfunction = _self.hyperfunction

    @staticmethod
    def patch_forward(self: "SNGemmaForCausalLM",
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
                                   cache_position: Optional[torch.LongTensor] = None,
                                   # [SambaNova] Integer index tensor of size 1, indicates current token generation position for cached inference
                                   last_token_index: Optional[torch.Tensor] = None
                                   ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, SNGemmaForCausalLM

        >>> model = SNGemmaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        consume_cache = use_cache and past_key_values is not None

        batch_size, seq_length = input_ids.shape

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            last_token_index=last_token_index)

        hidden_states = outputs[0]

        with self.hyperfunction.classifier(seq_length, consume_cache, self.training, reuse_last_id=True):
            logits = self.lm_head(hidden_states)
            if self.config.fp32_logits:
                logits = logits.float()

            loss = None
            if labels is not None:
                # [SambaNova] slicing is not efficient on RDU, so we expect the logits and labels are already shifted during
                # dataloader preprocessing. No need to shift again here.
                # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                # TODO: set --use-air-ce by default during both compile and runtime
                # [SambaNova] choose not to reduce by ??mean?? to allow setting correct gradients based on token types
                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits = logits.view(-1, self.config.vocab_size)
                shift_labels = labels.view(-1)
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
