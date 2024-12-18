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

from contextlib import contextmanager
from typing import Optional, Union

import torch


class fp32info:
    def __init__(self, min_value: float):
        self.min_value = min_value

    @property
    def min(self):
        return self.min_value


class FinfoPatch:
    def __init__(self, fp32_min_value: float, torch_finfo: 'torch.finfo'):
        self.fp32_min_value = fp32_min_value
        self.torch_finfo = torch_finfo

    def __call__(self, torch_dtype: Optional[torch.dtype] = None) -> Union['torch.finfo', fp32info]:
        if torch_dtype is None:
            torch_dtype = torch.get_default_dtype()

        if torch_dtype == torch.float32:
            info = fp32info(self.fp32_min_value)
        else:
            info = self.torch_finfo(torch_dtype)
        return info


# Refer to the docstring of finfo_float32_min_patch
MASK_MIN_VALUE = -10e10


@contextmanager
def finfo_float32_min_patch(val: float = MASK_MIN_VALUE):
    """
    Within the context, change the torch.finfo(torch.float32).min to -10e10.

    1. Model may call torch.tensor(float32.min) * 0. This gives 0 on both CPU and RDU which meets the IEEE standard.
    2. torch.tensor(float32.min).bfloat16() * 0 = -inf * 0 = NaN on both CPU and RDU which also meets the IEEE standard.
    3. However, Sambaflow mixed precision compilation support may cast float32 number to bfloat16.
    The bfloat16 conversion to float32.min results in -inf. Multiplying -if with 0 results in NaNs.

    To work with float32/bfloat16 mixed precision mode, this utility can be used when the following criteria apply:
    1. float32.min multipled by zero.
    2. Auto precision casting is enabled during compilation (which happens by default right now).

    Instead, we use -10e10 to replace float32.min. It works statistically fine for mask generation
    where float32.min is commonly used. The mask gets softmaxed later on so the effects of the value itself
    becomes insignificant numerically. We also left some rooms on the values here in case user does mask combination by adding them together. bfloat16.min + bfloat16.min can overflow to -inf. In addition to the above concern,
    -inf mask value is numerically unstable in SDPA due to segmentation of the softmax of continuous -inf that could introduce NaN as well:
     segmented_softmax([-inf, ..., -inf])
    = sum_recip_mul(exp(maxbias([-inf, ..., -inf])))
    = sum_recip_mul(exp([+inf, ..., +inf])) or sum_recip_mul(exp([Nan, ..., Nan])) (implementation dependent, but undesirable in any case)
    = sum_recip_mul([e^+inf, ..., e^+inf])
    = sum_recip_mul([+inf, ..., +inf])
    = [+inf,...,+inf] * recip(sum([+inf,...,+inf]))
    = [+inf,...,+inf] * recip(+inf)
    = [+inf,...,+inf] * 0
    = [+inf,...,+inf] or [Nan, Nan, ...Nan] (implementation dependent, but undesirable in any case)

    """
    original_finfo = torch.finfo
    torch.finfo = FinfoPatch(val, original_finfo)
    yield
    torch.finfo = original_finfo


def sn_patch_lazy_init_weights(self, module):
    """
    Patch lazy init of weights. This has a SambaFlow dependency and must be done only when is not JIT
    """
    from sambanova_modelzoo.models.lazy_init import sn_init_weights
    sn_init_weights(self, module)
