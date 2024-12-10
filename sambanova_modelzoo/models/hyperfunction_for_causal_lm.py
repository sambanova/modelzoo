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

from sambanova_modelzoo.models.config import SNPretrainedConfig


class HyperfunctionForCausalLM:
    def __init__(self, config: SNPretrainedConfig):
        self.config = config
        self._fuse_lm_head_with_sampling: bool = False

    @property
    def is_fuse_lm_head_with_sampling(self) -> bool:
        """
        Whether to fuse the language model head with sampling.
        """
        return self._fuse_lm_head_with_sampling

    @is_fuse_lm_head_with_sampling.setter
    def is_fuse_lm_head_with_sampling(self, value: bool) -> None:
        """
        Set whether to fuse the language model head with sampling.
        """
        self._fuse_lm_head_with_sampling = value

    @contextmanager
    def fuse_lm_head_with_sampling(self):
        """
        Context manager for annotating lm_head + sampling fusion. The lm_head is inside the modeling code. We traced it
        twice, once to slice out the lm_head only graph, the other one we trace it from a wrapper module CausalModelWithSampling
        that has both lm_head and sampling in separated modules. We use multigraph support to slice and combine them.

        We need different O1HD heuristics for these two sliced graphs; lm_head only and fused lm_head with sampling.
        """
        before = self._fuse_lm_head_with_sampling
        self.is_fuse_lm_head_with_sampling = True
        yield
        self.is_fuse_lm_head_with_sampling = before

    def MPMD_heuristic(self, heuristic_str):
        # return heuristic dict for MPMD(multiple program multiple data) heuristic hook
        return {
            "distribution": heuristic_str,
            "tiling": heuristic_str,
            "mapping": heuristic_str,
        }
