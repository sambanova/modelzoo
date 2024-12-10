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

from .modeling_mistral import SNMistralForCausalLM, SNMistralModel


class MistralTracer(CachedInferenceTracer, model=SNMistralForCausalLM):
    def __init__(self, config: PretrainedConfig, batch_size: int, token_gen_seq_length: int = 1):
        """
        Mistral class for generating dummy inputs for Samba tracing

        Args:
            config: Model config object
            batch_size: Batch size since we only support static tracing
            token_gen_seq_length: Length of token generation sequence length
        """
        assert token_gen_seq_length == 1, "Mistral only supports token_gen_seq_length=1"
        super().__init__(config, batch_size, token_gen_seq_length)

    @property
    def num_key_value_heads(self) -> int:
        return self.config.num_kv_heads

    def _get_kv_cache(self, traced_outputs: Tuple[Any, ...]) -> List[Tuple[SambaTensor, SambaTensor]]:
        """ Returns the list of KV cache for all layers from the traced output, used for cached inference """
        return [(traced_outputs[i * 2 + 1], traced_outputs[i * 2 + 2]) for i in range(self.config.num_hidden_layers)]


class E5MistralTracer(CachedInferenceTracer, model=SNMistralModel):
    def __init__(self, config: PretrainedConfig, batch_size: int, token_gen_seq_length: int = 1):
        """
        E5Mistral class for generating dummy inputs for Samba tracing. Note that the model is SNMistralModel instead of
        SNMistralForCausalLM.

        Args:
            config: Model config object
            batch_size: Batch size since we only support static tracing
        """
        super().__init__(config, batch_size, token_gen_seq_length)

    @property
    def num_key_value_heads(self) -> int:
        return self.config.num_kv_heads

    def _get_kv_cache(self, traced_outputs: Tuple[Any, ...]) -> List[Tuple[SambaTensor, SambaTensor]]:
        """ Returns the list of KV cache for all layers from the traced output, used for cached inference """
        return [(traced_outputs[i * 2 + 1], traced_outputs[i * 2 + 2]) for i in range(self.config.num_hidden_layers)]

    def get_token_gen_input_ids(self):
        raise NotImplementedError("E5Mistral does not have a token gen graph.")

    def get_attention_mask(self):
        raise NotImplementedError("E5Mistral does not have a token gen graph or any postprocess graphs.")

    def get_logits(self):
        raise NotImplementedError("E5Mistral does not produce logits.")

    def get_generated_tokens(self):
        raise NotImplementedError("E5Mistral does not generate tokens.")

    def get_generated_index(self):
        raise NotImplementedError("E5Mistral does not have a generated index.")

    def get_temperature(self):
        raise NotImplementedError("E5Mistral does not have temperature.")

    def get_top_k(self):
        raise NotImplementedError("E5Mistral does not have top_k.")

    def get_top_p(self):
        raise NotImplementedError("E5Mistral does not have top_p.")

    def get_postprocess_tracing_inputs(self):
        raise NotImplementedError("E5Mistral does not have any postprocess graphs.")

    def get_token_gen_tracing_inputs(self, cache_gen_outputs: Tuple[SambaTensor]):
        raise NotImplementedError("E5Mistral does not have a token gen graph.")

    def get_logits_output(self, outputs: Tuple[SambaTensor]):
        raise NotImplementedError("E5Mistral does not compute logits (the lm_head is not in SNMistralModel).")


class MistralPretrainTracer(PretrainTracer, model=SNMistralForCausalLM):
    pass
