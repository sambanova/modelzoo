# coding=utf-8
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
""" PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from sambanova_modelzoo.libs.nlp.core.directives import sdpa_directives
from sambanova_modelzoo.models.custom_ops import create_3d_attn_mask, triu_fill, upper_triangular_fill
from sambanova_modelzoo.models.llama.heuristics.hyperfunction_llama import LlamaHyperfunction
from sambanova_modelzoo.models.modeling_patch_utils import MASK_MIN_VALUE
from sambanova_modelzoo.models.modeling_utils import (apply_rotary_pos_emb, get_cos_sin_cache, get_position_ids,
                                     get_sliced_hidden_states, update_kv_cache, TensorCache)
from sambanova_modelzoo.models.utils import init_weights, logger_info, is_jit
from sambanova_modelzoo.models.directives import opfusion_id

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from sambanova_modelzoo.models.directives import add_aux_tensor

logger = logging.get_logger(__name__)
tensor_cache = TensorCache()

"""
This Llama patch is based off the Hugging Face model with transformers==4.31.0

The following are SambaNova specific prerequisites to successfully use the modified model:

1. For training, article attention is always used to restrict attention to be within an article. See create_3d_attn_mask.
   The attention_mask needs to be passed as Tuple[torch.Tensor, torch.Tensor] and gets expanded inside the model (on-chip).
2. For inference, always set use_cache=True, e.g. cached inference. The input_ids must be right padded for the first pass,
   e.g. cache generation pass. In the following pass (token generation), the input_ids must be of sequence_length 1.
   See CachedInferenceRuntime for how to prepare inputs to work with Hugging Face's model.generate function.
3. Prepare input_ids and labels to be sliced outside the model so that inputs[0:n] predicts label[n]. This improves 
   performance because shifting/slicing is currently not performant on RDU.
   The recommended shifting/slicing on inputs and labels is as follows:
        sentence = sentence[..., :-1]
        labels = labels[..., 1:]
   Or simply shift the labels and use CrossEntropyLoss to ignore the last token prediction, for example,
        labels = labels[..., 1:]
        labels = torch.cat((labels, torch.ones([labels.shape[0], 1], dtype=labels.dtype) * -100), dim=1)
    where -100 is the ignore_index for CrossEntropyLoss.
"""


def llama_mha(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              attention_mask: Optional[torch.Tensor],
              is_causal: bool,
              dropout_p: float,
              mixedp_attn: bool,
              use_gqa: bool = False):
    """
    Hugging Face version of normal MHA, e.g. LlamaAttention
    NOTE: Key should be already transposed for the first matmul.
    """
    if mixedp_attn:
        assert q.dtype == torch.bfloat16, "query's dtype must be bfloat16 if using mixed precision attention"
        assert k.dtype == torch.bfloat16, "key's dtype must be bfloat16 if using mixed precision attention"
        assert v.dtype == torch.bfloat16, "value's dtype must be bfloat16 if using mixed precision attention"

    if use_gqa:
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim = q.shape
        num_heads = num_kv_heads * num_kv_groups
    else:
        bsz, num_heads, q_len, head_dim = q.shape
    kv_seq_len = v.shape[-2]

    with opfusion_id('matmul'):
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
        # apply the block diagonal 3D mask, the masked_fill operation above does the upper triangular masking
        # already, which allows us to use a block diagonal 3D mask instead of having to construct a lower
        # triangular block diagonal mask
        attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)

    if is_causal:
        attn_weights = triu_fill(attn_weights, MASK_MIN_VALUE)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
        if use_gqa:
            attention_mask = attention_mask.unsqueeze(dim=1)
        attn_weights = attn_weights + attention_mask

    # [SambaNova] use mixedp softmax not float32
    # Do not enforce full fp32 attention, instead the compiler will determine whether to do fp32 or fp32/bf16 mixp
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)

    with opfusion_id('matmul_1'):
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


def llama_sdpa_switch(q: torch.Tensor,
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
    Args:
        q: Query states, input to SDPA.
        k: Key states, input to sdap.
        v: Value states, input to SDPA.
        attention_mask: If is a tuple, it will be viewed as a pair of collapsed 3D article mask, input to SDPA.
        is_causal: Whether to use causal masking, input to SDPA.
        dropout_p: Dropout percentage, input to SDPA.
        use_segmented_softmax_attn: Controls whether to use segmented softmax attention.
        seg_softmax_block_size: Controls the block size used in segmented softmax attention
        mixedp_attn: Controls the precision of the computation:
                     False: pure bf16/fp32 (depending on input dtype)
                     True: conservative mixp, first matmul is bf16->fp32, softmax maxbias is fp32, softmax exp is bf16
        key_transposed(Deprecated): If the key is already transposed for the first matmul.
        use_gqa: Whether to use group query attention

    Returns:
        A tuple of tensors. The first tensor is the output of SDPA, second tensor is attention score, the result of softmax in SDPA.
        The second tensor is garbage when `use_segmented_softmax_attn` is True.

    Restrictions:
        `is_causal` and non-3D `attention_mask` are mutually exclusive
         Assumes 4D q, k, v
    """

    has_attn_mask = attention_mask is not None
    if use_segmented_softmax_attn:
        # Reconstruct normal mask from collapsed 3D article attention mask, adding causal mask if needed.
        if key_transposed:  # air.SDPA transposes key inside
            k = k.transpose(2, 3)
        if type(attention_mask) is tuple and len(attention_mask) == 2:
            logger_info('Using 3D article attention mask in Llama2 Model.')
            attention_mask = create_3d_attn_mask(attention_mask, mixedp_attn, 0, MASK_MIN_VALUE)

        if attention_mask is not None:
            # make sure attention_mask has shape (bs, n_heads, q_ss, kv_ss)
            bs, n_heads, q_ss = q.shape[:3]
            kv_ss = k.shape[2]
            assert attention_mask.shape[-1] == kv_ss and attention_mask.shape[-2] == q_ss, \
                f"Last two dimensions of attention_mask should be q_seq_length and kv_seq_length. The mask has shape {attention_mask.shape} and expected last two dimensions is {(q_ss, kv_ss)}."
            if len(attention_mask.shape) == 3:  # (mask_bs, q_ss, kv_ss)
                attention_mask = torch.unsqueeze(attention_mask, 1)

            if mixedp_attn:
                attention_mask = attention_mask.to(torch.float32)

            if is_causal:
                # TODO: remove as right padding no longer needs attention_mask most of time
                attention_mask = triu_fill(attention_mask, MASK_MIN_VALUE)
                is_causal = False

        assert not (is_causal and has_attn_mask), "is_causal and attention mask are mutually exclusive in SDPA."

        with sdpa_directives({
                'sdp_mixed_p': mixedp_attn,
                'sdp_block_size': seg_softmax_block_size,
        }):
            logger_info('Using air SDPA in Llama2 Model.')
            with opfusion_id('sdpa'):
                o = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p, is_causal=is_causal)
            return o, o
    else:
        if key_transposed is not None:
            if not key_transposed:
                k = k.transpose(-1, -2)
        else:
            k = k.transpose(-1, -2)
        logger_info('Using patched regular attn in Llama2 Model.')
        return llama_mha(q, k, v, attention_mask, is_causal, dropout_p, mixedp_attn, use_gqa=use_gqa)


class LlamaMLPNamespace:
    @staticmethod
    def patch_forward(self, x):
        with opfusion_id('up_proj'):
                up_proj = self.up_proj(x)
    
        with opfusion_id('gate_proj'):
            gate_proj = self.gate_proj(x)
    
        result = self.act_fn(gate_proj) * up_proj
    
        with opfusion_id('down_proj'):
            down_proj = self.down_proj(result)
    
        return down_proj


class LlamaRMSNormNamespace:
    @staticmethod
    def patch_forward(self, hidden_states):
        # [SambaNova] instead of forcing to float32, we use mixed precision
        # hidden_states = hidden_states.to(torch.float32)
        if hasattr(self, 'config'):
            # in-memory patch overwrites config only, does not have fp32_ln in self
            fp32_ln = self.config.fp32_ln
        else:
            fp32_ln = self.fp32_ln
        hidden_states = hidden_states.to(torch.float32 if fp32_ln else torch.bfloat16)

        # We use a sum reduction followed by a division rather than a mean reduction.
        # This is necessary to permit RMS norm sharding on the reduction dimension,
        # since our compiler does not currently support sharding a mean on the reduction dimension. 
        variance = hidden_states.pow(2).sum(-1, keepdim=True) / hidden_states.shape[-1]
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class LlamaRotaryEmbeddingNamespace:
    @staticmethod
    def init_cos_sin_cache(self: 'LlamaRotaryEmbedding',
                           dim,
                           max_position_embeddings,
                           base,
                           device,
                           scaling_factor,
                           rope_type,
                           config
                           ):
     self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    @staticmethod
    def set_cos_sin_cache(self: 'LlamaRotaryEmbedding', seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        self.register_buffer("_cos_cached", self.attention_scaling * emb.cos().to(dtype), persistent=False)
        self.register_buffer("_sin_cached", self.attention_scaling * emb.sin().to(dtype), persistent=False)


    @staticmethod
    def patch_forward(self: 'LlamaRotaryEmbedding', x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # [SambaNova] slicing is not performant on RDU, use split instead, see below
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


class LlamaAttentionNamespace:
    @staticmethod
    def patch_forward(
            self: 'LlamaAttention',
            hidden_states: torch.Tensor,
            # [SambaNova] attention mask can be a tuple to represent 3D on-chip mask generation
            attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            last_token_index: Optional[torch.Tensor] = None,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        with opfusion_id('q_proj'):
            query_states = self.q_proj(hidden_states)

        with opfusion_id('k_proj'):
            key_states = self.k_proj(hidden_states)

        with opfusion_id('v_proj'):
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        with opfusion_id('v_proj_view'):
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value_provided = past_key_value is not None
        consume_cache = use_cache and past_key_value_provided

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value_provided:
            # [SambaNova] cached inference token-by-token generation after prompt
            past_key, past_value = past_key_value
            key_states, value_states = update_kv_cache(past_key, past_value, key_states, value_states, last_token_index)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
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
        with opfusion_id('reshape_k'):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output, attn_weights = llama_sdpa_switch(
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=not consume_cache,
            dropout_p=self.config.attention_dropout if self.training else 0,
            use_segmented_softmax_attn=self.config.use_segmented_softmax_attn,
            seg_softmax_block_size=self.config.seg_softmax_block_size,
            mixedp_attn=self.config.mixedp_attn,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            with opfusion_id('o_proj'):
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            with opfusion_id('o_proj'):
                attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayerNamespace:
    @staticmethod
    def patch_forward(
            self: 'LlamaDecoderLayer',
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
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
                Whether or not to return the attention tensors of all attention layers. See `attentions` under
                returned tensors for detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        # [SambaNova] used to enable fp32 residual accumulation
        if self.config.fp32_skip_add:
            residual = residual.float()

        with opfusion_id('input_layernorm'):
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

        with opfusion_id('post_attention_layernorm'):
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        with opfusion_id('gate_and_down_proj_add'):
            hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )
        return outputs


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    past_key_values_length = past_key_values_length if (past_key_values_length > 0) else tgt_len
    mask = torch.full((tgt_len, past_key_values_length), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, consume_cache):
    if attention_mask.dim() == 4:
        return attention_mask
    assert attention_mask.dim() == 2, f"The attention mask must be rank of 2, got {attention_mask.dim()}"
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if not consume_cache:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                          tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                                                                              combined_attention_mask)

    return combined_attention_mask


class SNLlamaModelNamespace:
    @staticmethod
    def add_hyperfunction_annotation(self: "SNLlamaModel", config, class_for_name: Type):
        # [SambaNova] For O1 heuristic annotation
        self.hyperfunction = LlamaHyperfunction(config, class_for_name)

    @staticmethod
    def patch_forward(
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
        # [SambaNova] Sanity check the attention mask.
        if self.training:
            assert not use_cache, "Please set the model to eval to do cached inference"
            err_msg = 'Attention mask must be a pair (article attention) or None (on-chip mask gen) during training'
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
                    f"max_seq_length must ({self.config.max_seq_length}) equal to the length of past_key_value input ({past_key_values_length}) in token_gen graph"


        with self.hyperfunction.embedding(seq_length, consume_cache, self.training):
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                # [SambaNova] generative inference does not need to pass position_ids because we use right padding
                position_ids = get_position_ids(batch_size, seq_length, last_token_index, device, consume_cache)

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            # FIXME: why the condition ?
            # [SambaNova] Only populate mask in either cached inference token-by-token generation or
            # TODO: or what?
            if consume_cache:
                # embed positions
                if attention_mask is None:
                    assert past_key_values is None
                    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)
                if not isinstance(attention_mask, tuple):
                    attention_mask = _prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length),
                                                                     inputs_embeds, past_key_values_length, consume_cache)
            hidden_states = inputs_embeds

            # [SambaNova] For cached inference, 1) cache_gen graph and seq_len need to be the same as input_ids length. 2)
            # token_gen graph, seq_len is the past_key_values_length. For training, seq_len is the same as iput_ids length.
            seq_len = max(seq_length, past_key_values_length)
            cos, sin = tensor_cache.get_or_create_tuple(
                f"cos_sin_tensors_{seq_len}",
                lambda: self.rotary_emb(hidden_states, position_ids, seq_len=seq_len)
            )
            add_aux_tensor(cos, f"aux_cos_{seq_len}")
            add_aux_tensor(sin, f"aux_sin_{seq_len}")
            cos, sin = get_cos_sin_cache(cos, sin, position_ids)

            # [SambaNova] In terms of compute graph, the float cast on inputs_embeds is redundant to the float cast within
            # the first decoder forward call where the input hidden_states, which is inputs_embeds here, is cast to float
            # again. This redundant cast is to ensure o1 plugin heuristics's hyperfunctions work for the first decoder.
            # Hyperfunctions require that the input metadata (includeing dtype) are the same across layers. Without this
            # cast, the first decoder layer's hidden state input may look like bfloat16 while the other decoder will always
            # be float. As a result, we might compile a different hyperfunction for the first decoder layer, doubling
            # compilation time and the amount of work in heuristics design.
            if self.config.fp32_skip_add:
                hidden_states = hidden_states.float()

        if self.gradient_checkpointing and self.training:
            if use_cache:
                if not is_jit():
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

            with self.hyperfunction.encoder_decoder(seq_length, consume_cache, self.training):
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
            # [SambaNova] Gather cannot be tiled along the gathering dimension.
            # Having this at the beginning of the classifier section allows the rest
            # of the classifier be pipelined without materalizing the full tensor
            if last_token_index is not None and not consume_cache:
                hidden_states = get_sliced_hidden_states(hidden_states, last_token_index)
            # [SambaNova]: Return hidden_states before norm for performant mapping through fusing norm, lm_head and sampling.
            last_hidden_states_before_norm = hidden_states
            with opfusion_id('norm'):
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


class SNLlamaForCausalLMNamespace:
    @staticmethod
    def add_hyperfunction_annotation(self: 'SNLlamaForCausalLM', config, class_for_name: Type):
        # [SambaNova] For O1 heuristic annotation
        self.hyperfunction = LlamaHyperfunction(config, class_for_name)

        # Overwrite and share the same hyperfunction object in the model as well
        self.model.hyperfunction = self.hyperfunction

    # TODO: the doc string for  sn_llama_for_causal_lm_forward is not complete.
    @staticmethod
    def patch_forward(
            self: 'SNLlamaForCausalLM',
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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is computed only for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, SNLlamaForCausalLM

        >>> model = SNLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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

        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]

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

        assert not (self.config.pretraining_tp > 1)
        with self.hyperfunction.classifier(seq_length, consume_cache, self.training, reuse_last_id=True):
            with opfusion_id('lm_head'):
                logits = self.lm_head(hidden_states)
            if self.config.fp32_logits:
                logits = logits.float()

            loss = None
            if labels is not None:
                # [SambaNova] slicing is not efficient on RDU. We expect the label to be the same as input_ids. Users need
                # to use external loss gradient calculation inside the application to disable the backward pass on the label
                # of the first token of each article. The loss gradient is the tensor you need to pass to the backward graph
                # to start backpropagation.

                # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                # [SambaNova] none reduction allows flexibility to do customized backpropagation
                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits = logits.view(-1, self.config.vocab_size)
                shift_labels = labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device, dtype=torch.long)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            if loss is not None:
                output = (loss, ) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


OVERWRITE_METHODS = {
    'LlamaForCausalLM': [('forward', SNLlamaForCausalLMNamespace.patch_forward)],
    'LlamaRMSNorm': [('forward', LlamaRMSNormNamespace.patch_forward)],
    'LlamaRotaryEmbedding': [('forward', LlamaRotaryEmbeddingNamespace.patch_forward)],
    'LlamaAttention': [('forward', LlamaAttentionNamespace.patch_forward)],
    'LlamaDecoderLayer': [('forward',LlamaDecoderLayerNamespace.patch_forward)],
    'LlamaModel': [('forward', SNLlamaModelNamespace.patch_forward)],
    'LlamaPreTrainedModel': [('_init_weights', init_weights)],
}

# A fake version to disable in-memory patch
TRANSFORMERS_VERSION = '-1'
