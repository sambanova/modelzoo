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

from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from sambanova_modelzoo.models.modeling_patch_utils import MASK_MIN_VALUE, finfo_float32_min_patch
from sambanova_modelzoo.models.utils import is_jit

### Samba/JIT wrapper functions. JIT is currently a SambaNova internal initiative.

Tensor = Union['SambaTensor', torch.Tensor]


def create_3d_attn_mask(
        attention_mask_collapsed: Tuple[Tensor, Tensor],
        mixedp_attn: bool = False,
        attend_value: float = 1.0,
        ignore_value: float = 0.0,
) -> Tensor:
    """Selector for creating 3d attention mask between JIT and Samba"""
    if not is_jit():
        from sambanova_modelzoo.models.custom_samba_ops import create_3d_attn_mask as c3am
        return c3am(attention_mask_collapsed, mixedp_attn, attend_value, ignore_value)
    raise NotImplementedError("Not implemented for JIT yet")


def upper_triangular_fill(x, fill_value=-10e10, mixedp_attn=False):
    """Selector for upper triangular fill between JIT and Samba"""
    if not is_jit():
        from sambanova_modelzoo.models.custom_samba_ops import upper_triangular_fill as utf
        return utf(x, fill_value, mixedp_attn)
    raise NotImplementedError("Not implemented for JIT yet")


def triu_fill(input, value: float):
    """Selector for masked fill between JIT and Samba"""
    if not is_jit():
        import sambaflow.samba as samba
        return samba.triu_fill(input, value)
    return torch.ops.torch_rdu.triufill_air(input, value)


def sliding_window_fill(x: Tensor,
                        fill_value: float = MASK_MIN_VALUE,
                        sliding_window_size: int = 4096,
                        mixedp_attn: bool = False):
    """Selector for sliding window fill between JIT and Samba"""
    if not is_jit():
        from sambanova_modelzoo.models.custom_samba_ops import sliding_window_fill as swf
        return swf(x, fill_value, sliding_window_size, mixedp_attn)
    raise NotImplementedError("Not implemented for JIT yet")


def scatter(operand: Tensor,
            update: Tensor,
            start_indices: Tensor,
            scatter_dims: List[int],
            rmw_op: str = "kUpdate",
            unique_indices: bool = False,
            batched_dims: int = 0) -> Tensor:
    """Selector for scatter op between JIT and Samba"""
    if is_jit():
        assert not unique_indices, "unique_indices must be False when running through JIT"
        return torch.ops.torch_rdu.scatter_air(operand, update, start_indices, scatter_dims, rmw_op, batched_dims)
    else:
        import sambaflow.samba.sn_private as sp
        return sp.sn_scatter(operand,
                             update,
                             start_indices,
                             scatter_dims,
                             rmw_op=rmw_op,
                             unique_indices=unique_indices,
                             batched_dims=batched_dims)


def gather(input_tensor: Tensor,
           start_indices: Tensor,
           gather_dims: List[int],
           gather_lengths: List[int],
           batched_dims: int = 0) -> Tensor:
    """Selector for gather op between JIT and Samba"""
    if is_jit():
        return torch.ops.torch_rdu.gather_air(input_tensor, start_indices, gather_dims, gather_lengths, batched_dims)
    else:
        import sambaflow.samba.sn_private as sp
        return sp.sn_gather(input_tensor, start_indices, gather_dims, gather_lengths, batched_dims=batched_dims)


def topk(routing_weights: Tensor, top_k: int, cat: bool = True) -> Tuple[List[Tensor], List[Tensor]]:
    """SambaNova doesn't support torch.topk for tensors of shape [M, N] where M, N > 1. This function replaces
    instances of torch.topk in Mixtral with operations supported on the RDU. This function assumes that :attr:`top_k`
    is small, the input tensor is the tensor of routing weights, and that the dim is -1.

    Args:
        routing_weights: the tensor of expert weights for each token
        top_k: the k in "top-k"
        cat: whether to torch.cat the results at the end. During inference, the cache generation and token generation
            graphs loop over selected experts, so cat=False.

    Returns:
        topk_routing_weights: the :attr:`top_k` values of :attr:`routing_weights` on dim=-1
        selected_experts: the indices of the top-k values.
    """
    new_routing_weights = []
    selected_experts = []
    for i in range(top_k):
        max_val, i_max = torch.max(routing_weights, dim=-1, keepdim=True)
        new_routing_weights.append(max_val)
        selected_experts.append(i_max)
        if i != top_k - 1:
            with finfo_float32_min_patch():
                scatter_indices = torch.cat(
                    [torch.arange(routing_weights.shape[0]).reshape(-1, 1).to(torch.int32),
                     i_max.to(torch.int32)],
                    dim=-1)
                scatter_update = torch.tensor([[[torch.finfo(torch.float).min]]
                                               ]).repeat(routing_weights.shape[0], 1, 1).to(routing_weights.dtype)
                routing_weights = scatter(routing_weights, scatter_update, scatter_indices, [0, 1])
    if cat:
        return torch.cat(new_routing_weights, dim=-1), torch.cat(selected_experts, dim=-1)
    else:
        return new_routing_weights, selected_experts


def groupby(tensor: Tensor,
            num_bins: int = 32,
            capacity: int = 1,
            samples_to_reset: int = 1,
            with_hist_buffer: bool = False,
            no_overflow_bin: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Refer to samba.groupby
    """
    if is_jit():
        raise NotImplementedError()
    import sambaflow.samba as samba
    return samba.groupby(tensor,
                         num_bins=num_bins,
                         capacity=capacity,
                         samples_to_reset=samples_to_reset,
                         with_hist_buffer=with_hist_buffer,
                         no_overflow_bin=no_overflow_bin)


def addmm(input: Tensor,
          mat1: Tensor,
          mat2: Tensor,
          *,
          beta: Optional[Number] = 1,
          alpha: Optional[Number] = 1,
          out: Optional[Tensor] = None,
          is_transposed: Optional[bool] = False) -> Tensor:
    """
    Refer to samba.addmm
    """
    if is_jit():
        raise NotImplementedError()
    import sambaflow.samba as samba
    return samba.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out, is_transposed=is_transposed)


def sn_multinomial(probs: Tensor, pre_generated_randoms: Tensor) -> Tensor:
    """Refer to sn_multinomial"""
    if is_jit():
        raise NotImplementedError("Not implemented for JIT yet")
    from sambaflow.samba.sn_private import sn_multinomial
    return sn_multinomial(probs, pre_generated_randoms)


def sn_zipmapreduce(func: Callable[[Dict[str, Any], Iterable[Tensor]], Tensor],
                    inputs: List[Tensor],
                    reduce_func: Optional[str] = None,
                    reduce_dim: Optional[Union[int, List[int]]] = None,
                    stoc_accum: Optional[bool] = False) -> Tensor:
    """Refer to sn_zipmapreduce"""
    if is_jit():
        raise NotImplementedError("Not implemented for JIT yet")
    from sambaflow.samba.functional.stir import sn_zipmapreduce
    return sn_zipmapreduce(func=func,
                           inputs=inputs,
                           reduce_func=reduce_func,
                           reduce_dim=reduce_dim,
                           stoc_accum=stoc_accum)


def sn_imm(input: Union[float, int], dtype: torch.dtype) -> Tensor:
    """Refer to sn_imm"""
    if is_jit():
        raise NotImplementedError("Not implemented for JIT yet")
    from sambaflow.samba.functional.stir import sn_imm
    return sn_imm(input=input, dtype=dtype)


def sn_iteridx(attrs: dict, dim: int, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Refer to sn_iteridx"""
    if is_jit():
        raise NotImplementedError("Not implemented for JIT yet")
    from sambaflow.samba.functional.stir import sn_iteridx
    return sn_iteridx(attrs=attrs, dim=dim, dtype=dtype)


def sn_select(cond: Tensor, true_val: Tensor, false_val: Tensor) -> Tensor:
    """Refer to sn_select"""
    if is_jit():
        raise NotImplementedError("Not implemented for JIT yet")
    from sambaflow.samba.functional.stir import sn_select
    return sn_select(cond=cond, true_val=true_val, false_val=false_val)
