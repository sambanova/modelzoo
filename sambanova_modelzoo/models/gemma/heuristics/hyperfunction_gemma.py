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

from typing import ContextManager, Type

from sambanova_modelzoo.models.config import SNPretrainedConfig
from sambanova_modelzoo.models.directives import op_fusion
from sambanova_modelzoo.models.hyperfunction_for_causal_lm import HyperfunctionForCausalLM


class GemmaHyperfunction(HyperfunctionForCausalLM):
    """
    This class provides directive context manager for O1 heuristic annotation to define the
    hyperfunction boundary.
    """
    def __init__(self, config: SNPretrainedConfig, class_for_name: Type):
        super().__init__(config)
        self.class_name = class_for_name.__name__.lower()

    def embedding(self, input_seq_length: int, consume_cache: bool, is_training: bool,
                  reuse_last_id: bool = False) -> ContextManager:
        """ Hyperfunction for embedding """
        if is_training:
            heuristic = ""
            plugins = ""
        else:
            heuristic_str = "kEmbedding"
            heuristic = self.MPMD_heuristic(heuristic_str)
            plugins = ["libEmbeddingPlugins.so"]

        return op_fusion(
            func_name=f'emb_mask_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}',
            heuristic=heuristic,
            heuristic_params={
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
            },
            plugins=plugins,
            reuse_last_id=reuse_last_id,
            user_annotated=self.config.use_plugin_heuristics)

    def encoder_decoder(self, input_seq_length: int, consume_cache: bool,
                        reuse_last_id: bool = False) -> ContextManager:
        """
        Hyperfunction for encoder/decoder
        """

        heuristic_str = "kGemma_Encoder"
        heuristic = self.MPMD_heuristic(heuristic_str)
        plugins = ["libGemmaEncoderHook.so"]
        return op_fusion(
            func_name=f'encoder_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}',
            heuristic=heuristic,
            heuristic_params={
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
                "class_name": self.class_name,
            },
            plugins=plugins,
            reuse_last_id=reuse_last_id,
            user_annotated=self.config.use_plugin_heuristics)

    def classifier(self, input_seq_length: int, consume_cache: bool, is_training: bool,
                   reuse_last_id: bool = False) -> ContextManager:
        """ Hyperfunction for classifier """

        if is_training:
            heuristic = ""
            params = None
            plugins = ""
        else:
            heuristic_str = "kGemma_Classifier"
            heuristic = self.MPMD_heuristic(heuristic_str)
            params = {
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
                "class_name": self.class_name
            }
            plugins = ["libGemmaClassifierHook.so"]

        return op_fusion(func_name=f'cls_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}',
                         heuristic=heuristic,
                         heuristic_params=params,
                         plugins=plugins,
                         reuse_last_id=reuse_last_id,
                         user_annotated=self.config.use_plugin_heuristics)
