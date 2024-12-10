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


class LlamaHyperfunction(HyperfunctionForCausalLM):
    """
    This class provides directive context manager for O1 heuristic annotation to define the
    hyperfunction boundary.
    """
    def __init__(self, config: SNPretrainedConfig, class_for_name: Type):
        super().__init__(config)
        self.class_name = class_for_name.__name__.lower()

        # for the case of fused lm_head+postprocess graph, record the hyperfunction info to use in the postprocess
        # module
        self.fused_cls_postprocess_hyperfunction = None

    def embedding(self, input_seq_length: int, consume_cache: bool, is_training: bool,
                  reuse_last_id: bool = False) -> ContextManager:
        """ Hyperfunction for embedding """

        heuristic, params, plugins = None, None, None
        func_name = f'emb_ss_{input_seq_length}_{self.config.model_type}'

        if self.config.use_plugin_heuristics and is_training:
            # For training, we use the MPMD heuristic for embedding.
            heuristic_str = "kLlama3_training_Embedding"
            heuristic = self.MPMD_heuristic(heuristic_str)
            plugins = "libLlama3TrainingEmbeddingHook.so"
            params = {
                "max_seq_length": self.config.max_seq_length,
                "vocab_size": self.config.vocab_size,
            }

        elif self.config.use_plugin_heuristics and not is_training:
            if self.config.run_early_tp:
                heuristic = "kLlama3_Embedding_SPMD"
                plugins = "libLlama3SPMD_EmbeddingHook.so"
            else:
                heuristic_str = "kEmbedding"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libEmbeddingPlugins.so"]

            func_name = f'emb_mask_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}'
            params = {
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
            }

        return op_fusion(func_name=func_name,
                         heuristic=heuristic,
                         heuristic_params=params,
                         plugins=plugins,
                         reuse_last_id=reuse_last_id,
                         user_annotated=self.config.use_plugin_heuristics)

    def encoder_decoder(self,
                        input_seq_length: int,
                        consume_cache: bool,
                        is_training: bool,
                        reuse_last_id: bool = False) -> ContextManager:
        """
        Hyperfunction for encoder/decoder
        """

        heuristic, params, plugins = None, None, None

        gqa = self.config.num_attention_heads != self.config.num_key_value_heads

        hidden_size = self.config.hidden_size
        use_sdpa = self.config.use_segmented_softmax_attn
        func_name = f'encoder_ss_{input_seq_length}_{self.config.model_type}'

        if self.config.use_plugin_heuristics and is_training:
            if (gqa, hidden_size, use_sdpa) == (True, 4096, True):
                heuristic_str = "kLlama3_training_8B_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama3Training8bEncoderHook.so"]
            params = {
                "max_seq_length": self.config.max_seq_length,
                "class_name":
                self.config.architectures[0].lower() if self.config.architectures else "snllamaforcausallm",
                "vocab_size": self.config.vocab_size,
            }

        elif self.config.use_plugin_heuristics and not is_training:
            if (gqa, hidden_size, use_sdpa) == (True, 4096, False) and self.config.run_early_tp:
                heuristic = {
                    "distribution": "kLlama3_8B_Encoder_SPMD",
                    "tiling": "kLlama3_Encoder_SPMD",
                    "mapping": "kLlama3_8B_Encoder_SPMD",
                    "multi_socket_distribution": "kLlama3_Encoder_SPMD"
                }
                plugins = [
                    "libLlama3SPMD_8B_EncoderMappingHook.so", "libLlama3SPMD_EncoderTilingHook.so",
                    "libLlama3SPMD_8B_EncoderDistributionHook.so", "libLlama3SPMD_EncoderMultiSocketDistributionHook.so"
                ]
            elif (gqa, hidden_size, use_sdpa) == (True, 4096, True):
                heuristic_str = "kLlama3_8B_Sdpa_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama3SdpaEncoderHook.so"]
            elif (gqa, hidden_size, use_sdpa) == (True, 4096, False):
                heuristic_str = "kLlama3_8B_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama3EncoderHook.so"]
            elif (gqa, hidden_size, use_sdpa) == (True, 8192, False) and not self.config.run_early_tp:
                heuristic_str = "kLlama3_70B_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama3_70bEncoderHook.so"]
            elif (gqa, hidden_size, use_sdpa) == (True, 8192, False) and self.config.run_early_tp:
                heuristic = {
                    "distribution": "kLlama3_70B_Encoder_SPMD",
                    "tiling": "kLlama3_Encoder_SPMD",
                    "mapping": "kLlama3_70B_Encoder_SPMD",
                    "multi_socket_distribution": "kLlama3_Encoder_SPMD"
                }
                plugins = [
                    "libLlama3SPMD_70B_EncoderMappingHook.so", "libLlama3SPMD_EncoderTilingHook.so",
                    "libLlama3SPMD_70B_EncoderDistributionHook.so",
                    "libLlama3SPMD_EncoderMultiSocketDistributionHook.so"
                ]
            elif (gqa, hidden_size, use_sdpa) == (True, 16384, False) and self.config.run_early_tp:
                heuristic = {
                    "distribution": "kLlama3_405B_Encoder_SPMD",
                    "tiling": "kLlama3_405B_Encoder_SPMD",
                    "mapping": "kLlama3_405B_Encoder_SPMD",
                    "multi_socket_distribution": "kLlama3_Encoder_SPMD"
                }
                plugins = [
                    "libLlama3SPMD_405B_EncoderMappingHook.so", "libLlama3SPMD_405B_EncoderTilingHook.so",
                    "libLlama3SPMD_405B_EncoderDistributionHook.so",
                    "libLlama3SPMD_EncoderMultiSocketDistributionHook.so"
                ]
            elif (gqa, hidden_size, use_sdpa) == (True, 7168, False):
                heuristic_str = "kDeepseek_33B_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libDeepseek33BEncoderHook.so"]
            elif (gqa, hidden_size, use_sdpa) == (True, 7168, True):
                heuristic_str = "kDeepseek_33B_Sdpa_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libDeepseek33BSdpaEncoderHook.so"]
            elif (gqa, use_sdpa) == (False, False):
                heuristic_str = "kLlama2_7B_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama2EncoderHook.so"]
            elif (gqa, use_sdpa) == (False, True):
                heuristic_str = "kLlama2_Sdpa_Encoder"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama2SdpaEncoderHook.so"]
            else:
                raise ValueError(f"Cannot find O1HD Encoder Heuristic for config {self.config}")

            func_name = f'encoder_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}'
            params = {
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
                "class_name": self.class_name
            }

        return op_fusion(func_name=func_name,
                         heuristic=heuristic,
                         heuristic_params=params,
                         plugins=plugins,
                         reuse_last_id=reuse_last_id,
                         user_annotated=self.config.use_plugin_heuristics)

    def classifier(self, input_seq_length: int, consume_cache: bool, is_training: bool,
                   reuse_last_id: bool = False) -> ContextManager:
        """ Hyperfunction for classifier """

        hidden_size = self.config.hidden_size
        heuristic, params, plugins = None, None, None
        func_name = f'cls_ss_{input_seq_length}_{self.config.model_type}'

        if self.config.use_plugin_heuristics and is_training:
            heuristic_str = "kLlama3_training_Classifier"
            heuristic = self.MPMD_heuristic(heuristic_str)
            plugins = ["libLlama3TrainingClassifierHook.so"]
            params = {
                "max_seq_length": self.config.max_seq_length,
                "class_name":
                self.config.architectures[0].lower() if self.config.architectures else "snllamaforcausallm",
                "vocab_size": self.config.vocab_size,
            }
        elif self.config.use_plugin_heuristics and not is_training:
            if self.config.run_early_tp:
                if hidden_size == 16384:
                    heuristic = "kLlama3_405B_Classifier_SPMD"
                    plugins = ["libLlama3SPMD_405B_ClassifierHook.so"]
                else:
                    heuristic = "kLlama3_Classifier_SPMD"
                    plugins = ["libLlama3SPMD_ClassifierHook.so"]
            else:
                heuristic_str = "kLlama3_8B_Classifier"
                heuristic = self.MPMD_heuristic(heuristic_str)
                plugins = ["libLlama3ClassifierHook.so"]

            params = {
                "input_seq_length": input_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "consume_cache": consume_cache,
                "class_name": self.class_name
            }
            func_name = f'cls_consume_cache_{consume_cache}_ss_{input_seq_length}_{self.config.model_type}'

        # self._fuse_lm_head_with_sampling is set in the fuse_lm_head_with_sampling context manager when the lm_head and
        # postprocess modules are fused into one graph
        if self._fuse_lm_head_with_sampling:
            # store the op_fusion annotation as an instance attribute to be used later in postprocess module
            self.fused_cls_postprocess_hyperfunction = op_fusion(func_name=f'{func_name}_postprocess_fused',
                                                                 heuristic=heuristic,
                                                                 heuristic_params=params,
                                                                 plugins=plugins,
                                                                 reuse_last_id=reuse_last_id,
                                                                 user_annotated=self.config.use_plugin_heuristics)
            return self.fused_cls_postprocess_hyperfunction
        else:
            return op_fusion(func_name=func_name,
                             heuristic=heuristic,
                             heuristic_params=params,
                             plugins=plugins,
                             reuse_last_id=reuse_last_id,
                             user_annotated=self.config.use_plugin_heuristics)
