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

from typing import Optional

import torch
import torch.nn as nn

from sambaflow.samba.lazy_param import RNGProvider, RNGProviderFactory
from sambaflow.samba.nn.parameter import SambaParameter

# TODO: At top of this file or in doc, explain what lazy init does.


# [SambaNova] to lazy initialize weights
class NormalWeightInit:
    def __init__(self, mean: float, std: float, rng_provider: RNGProvider):
        self.mean = mean
        self.std = std
        self.rng_provider = rng_provider

    """ Normal distributed weight materializer """
    def __call__(self, p: SambaParameter) -> torch.Tensor:
        return self.rng_provider().normal(self.mean, self.std, p.shape, dtype=p.dtype)


class ConstantInit:
    def __init__(self, constant):
        self.constant = constant

    """ Constant weight materializer """
    def __call__(self, p: SambaParameter) -> torch.Tensor:
        t = torch.empty(p.shape, dtype=p.dtype)
        t.fill_(self.constant)
        return t


class EmbeddingWeightInit:
    def __init__(self, padding_idx: Optional[int], std: float, rng_provider: RNGProvider):
        self.padding_idx = padding_idx
        self.std = std
        self.rng_provider = rng_provider

    """ Embedding weight materializer """
    def __call__(self, p: SambaParameter) -> torch.Tensor:
        t = self.rng_provider().normal(0.0, self.std, p.shape, dtype=p.dtype)
        if self.padding_idx is not None:
            t[self.padding_idx].zero_()
        return t


def normal_lazy_param(param: torch.Tensor, mean: float, std: float, rng_provider: RNGProvider) -> SambaParameter:
    """ Returns a lazy SambaParameter with zero mean and a variable std """
    return SambaParameter(shape=param.shape,
                          dtype=param.dtype,
                          materializer=NormalWeightInit(mean, std, rng_provider),
                          requires_grad=param.requires_grad)


def constant_lazy_param(param: torch.Tensor, v: float) -> SambaParameter:
    """ Returns a lazy SambaParameter with constant v """
    return SambaParameter(shape=param.shape,
                          dtype=param.dtype,
                          materializer=ConstantInit(v),
                          requires_grad=param.requires_grad)


def zero_lazy_param(param: torch.Tensor) -> SambaParameter:
    """ Returns a lazy SambaParameter with constant 0 """
    return constant_lazy_param(param, 0)


def one_lazy_param(param: torch.Tensor) -> SambaParameter:
    """ Returns a lazy SambaParameter with constant 1 """
    return constant_lazy_param(param, 1)


def embedding_lazy_param(param: torch.Tensor, padding_idx: Optional[int], std: float,
                         rng_provider: RNGProvider) -> SambaParameter:
    return SambaParameter(shape=param.shape,
                          dtype=param.dtype,
                          materializer=EmbeddingWeightInit(padding_idx, std, rng_provider),
                          requires_grad=param.requires_grad)


def sn_init_weights(self, module: torch.nn.Module):
    rng_factory = RNGProviderFactory()
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        # [SambaNova] using lazy init can avoid initializing the model before loading the checkpoint.
        # Even if training from scratch, SambaNova's lazy init leverages multithreading for a significant speedup
        if self.config.lazy_init:
            module.weight = normal_lazy_param(module.weight, 0.0, std, rng_factory.build_provider())
        else:
            module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            if self.config.lazy_init:
                module.bias = zero_lazy_param(module.bias)
            else:
                module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        if self.config.lazy_init:
            module.weight = embedding_lazy_param(module.weight, module.padding_idx, std, rng_factory.build_provider())
        else:
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        if self.config.lazy_init:
            module.bias = zero_lazy_param(module.bias)
            module.weight = one_lazy_param(module.weight)
        else:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
