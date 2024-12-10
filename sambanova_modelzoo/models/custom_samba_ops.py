# Copyright 2023-2024 SambaNova Systems, Inc.
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

from typing import Tuple

import torch
from sambanova_modelzoo.models.modeling_patch_utils import MASK_MIN_VALUE

import sambaflow
from sambaflow.samba.functional.stir import sn_imm, sn_iteridx, sn_select, sn_zipmapreduce
from sambaflow.samba.utils import SNType


# [SambaNova] 3D attention mask is useful to force attention to only attend within an article when there are multiple
# articles collapsed into a single batch of inputs. See reference to this article:
# OPT-IML : Scaling Language Model Instruction Meta Learning through the Lens of Generalization
# (https://arxiv.org/pdf/2212.12017.pdf section 3.2)
# It also uses on-chip mask gen to save memory bandwidth
def create_3d_attn_mask(
        attention_mask_collapsed: Tuple[torch.Tensor, torch.Tensor],
        mixedp_attn: bool = False,
        attend_value: float = 1.0,
        ignore_value: float = 0.0,
) -> torch.Tensor:
    """Create the full 3D article attention mask from the collapsed representation of the attention mask.
    3D masking uses on-chip mask generation to speed up performance. In order to use 3D masking, we expect the user
    1) collapses multiple articles into a long one for inputs 2) uses causual attentions instead of attenting future
    tokens.

    We create the 3D article attention mask in the model from the collapsed representation (instead of
    directly passing the full 3D attention mask) because the 3D attention mask becomes too big for long sequences.
    For an 8k sequence size, the 3D attention mask is size [batch_size, 8k, 8k]. The collapsed representation is
    much smaller, it's a tuple of two tensors, each of size [batch_size, 8k].  For 8k sequence size, the 3D attention
    mask is so big that we can't even store it on chip SRAM.  Hence, it makes sense to construct the mask on chip so
    we can do tiling to make sure we never materialize the entire mask at any time.

    How is the 3D attention mask constructed?  Let's say assume the input sequence is:
    'How are you <end of text> I'm good <endoftext>' and each word is a token.  In this situation, we want the
    article attention mask to be:
    1 0 0 0 0 0 0
    1 1 0 0 0 0 0
    1 1 1 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 1 0 0
    0 0 0 0 1 1 0
    0 0 0 0 0 0 0

    A `1` at position (i,j) means the ith token should attend to the jth token, and a `0` at that position would mean
    that the ith token should ignore the jth token during attention.

    To construct this, our collapsed attention masks are two tensors which look like:
    mask_1 = [1 1 1 0 2 2 0] and mask_2 = [1 1 1 -1 2 2 -1]. Also, we set ignore_value = 0 and attend_value = 1
    to set the values in the mask. After doing mask_1 == mask_2.t() (broadcast equal)
    we get:
    1 1 1 0 0 0 0
    1 1 1 0 0 0 0
    1 1 1 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 1 1 0
    0 0 0 0 1 1 0
    0 0 0 0 0 0 0
    We apply this mask to the input, and then apply a lower triangular diagonal masked fill to the input and
    effectively replicate applying the regular 3D attention mask to the input.

    Args:
        attention_mask_collapsed: The collapsed representation of the 3D attention mask.
        mixedp_attn: Flag controlling the dtype of the output mask. FP32 if True, BF16 if False.
        attend_value: The value at positions in the mask where attention is wanted.
        ignore_value: The value at positions in the mask where attention is ignored.

    Returns:
        The full size 3D attention mask of shape [batch_size, 1, sequence_length, sequence_length]
    """

    imm_type = torch.float32 if mixedp_attn else torch.bfloat16

    def func(i, mask, mask_t):
        return sn_select(mask == mask_t, sn_imm(attend_value, imm_type), sn_imm(ignore_value, imm_type))

    # attention_mask_collapsed could potentially be a generator, need to turn into list before index access
    mask_tmp = list(attention_mask_collapsed)
    # 2nd element of attention_mask_collapsed should already be transposed on host.
    return sn_zipmapreduce(func, [mask_tmp[0].to(imm_type), mask_tmp[1].to(imm_type)])


def upper_triangular_fill(x, fill_value=-10e10, mixedp_attn=False):
    """Fills upper triangle tensor (excluding main diagonal) with fill_value.
    Commonly used to do causal masking.

    Assumes 4D tensor. Diagonal viewed from dim 2 and 3.
    - If `mixedp_attn` is False, filled value will be in bf16.
    - If `mixedp_attn` is True, filled value will be in fp32.

    Restriction:
    When `mixedp_attn=False`, size of dim 2 and 3 of `x` should not be larger than 2**16.
    When `mixedp_attn=True`, size of dim 2 and 3 of `x` should not be larger than 2**31.
    """

    imm_type = torch.float32 if mixedp_attn else torch.bfloat16
    idx_type = SNType.INT32 if mixedp_attn else SNType.UINT16

    if not sambaflow.samba.session.use_static_functional():
        # for torch-in/torch-out behavior
        idx_type = SNType.to_torch_reference_type(idx_type)

    def triu_fill(attrs, x):
        dim_l = sn_iteridx(dim=2, attrs=attrs, dtype=idx_type)
        dim_r = sn_iteridx(dim=3, attrs=attrs, dtype=idx_type)
        condition = torch.ge(dim_l, dim_r)
        result = sn_select(condition, x, sn_imm(fill_value, imm_type))
        return result

    return sn_zipmapreduce(triu_fill, [x])


def sliding_window_fill(x: torch.Tensor,
                        fill_value: float = MASK_MIN_VALUE,
                        sliding_window_size: int = 4096,
                        mixedp_attn: bool = False):
    """
    This function fills a tensor with a sliding window pattern.

    Args:
        x (torch.Tensor): The input tensor with shape (bs, n_heads, q_ss, kv_ss).
        fill_value (float, optional): The value to fill the tensor with. Defaults to MASK_MIN_VALUE.
        sliding_window_size (int, optional): The size of the sliding window. Defaults to 4096.
        mixedp_attn (bool, optional): If True, uses float32 on fill value, else bfloat16. Defaults to False.

    Returns:
        torch.Tensor: The tensor with sliding window pattern with shape (bs, n_heads, q_ss, kv_ss).
    """
    assert len(x.shape) == 4
    imm_type = torch.float32 if mixedp_attn else torch.bfloat16
    idx_type = SNType.INT32 if mixedp_attn else SNType.UINT16

    if not sambaflow.samba.session.use_static_functional():
        # for torch-in/torch-out behavior
        idx_type = SNType.to_torch_reference_type(idx_type)

    def sliding_window_fill(attrs, x):
        dim_l = sn_iteridx(dim=2, attrs=attrs, dtype=idx_type)
        dim_r = sn_iteridx(dim=3, attrs=attrs, dtype=idx_type)
        and_left = torch.le(dim_l - dim_r, sn_imm(sliding_window_size, idx_type))
        and_right = torch.ge(dim_l, dim_r)
        condition = torch.bitwise_and(and_left, and_right)
        result = sn_select(condition, x, sn_imm(fill_value, imm_type))
        return result

    return sn_zipmapreduce(sliding_window_fill, [x])
