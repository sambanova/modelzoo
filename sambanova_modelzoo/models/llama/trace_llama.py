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

from typing import Any, List, Tuple

from sambanova_modelzoo.libs.nlp.core.clm_tracer import CachedInferenceTracer, PretrainTracer
from sambanova_modelzoo.models.config import PretrainedConfig

from sambaflow.samba import SambaTensor

from .modeling_llama import SNLlamaForCausalLM


class LlamaTracer(CachedInferenceTracer, model=SNLlamaForCausalLM):
    def __init__(self, config: PretrainedConfig, batch_size: int, token_gen_seq_length: int = 1):
        """
        Llama class for generating dummy inputs for Samba tracing
        
        Args:
            config: Model config object
            batch_size: Batch size since we only support static tracing
            token_gen_seq_length: Length of token generation sequence length
        """
        super().__init__(config, batch_size, token_gen_seq_length)

    @property
    def num_key_value_heads(self) -> int:
        return self.config.num_key_value_heads

    def _get_kv_cache(self, traced_outputs: Tuple[Any, ...]) -> List[Tuple[SambaTensor, SambaTensor]]:
        """ Returns the list of KV cache for all layers from the traced output. Used for cached inference """
        return [(traced_outputs[i * 2 + 1], traced_outputs[i * 2 + 2]) for i in range(self.config.num_hidden_layers)]


class LLamaPretrainTracer(PretrainTracer, model=SNLlamaForCausalLM):
    pass
