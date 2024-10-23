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

from typing import Dict, Type

from sambanova_modelzoo.libs.nlp.core.clm_runtime import CachedInferenceRuntime
from sambanova_modelzoo.models.config import SNPretrainedConfig
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformerPlugin
from sambanova_modelzoo.models.configuration_validator import SNConfigValidatorPlugin
from sambanova_modelzoo.models.mistral.configuration_mistral import SNMistralConfig
from sambanova_modelzoo.models.mistral.modeling_mistral import SNMistralForCausalLM, SNMistralModel
from sambanova_modelzoo.models.model_loader import ModelLoaderPlugin
from transformers import AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralModel


class MistralConfigurationTransformer(ConfigurationTransformerPlugin):
    def get_source_conversion_type(self) -> Type[PretrainedConfig]:
        return MistralConfig

    def get_target_conversion_type(self) -> Type[SNPretrainedConfig]:
        return SNMistralConfig

    def get_architectures_transform_map(self) -> Dict[Type[PreTrainedModel], Type[PreTrainedModel]]:
        return {
            MistralForCausalLM: SNMistralForCausalLM,
            MistralModel: SNMistralModel
        }


class MistralConfigValidator(SNConfigValidatorPlugin):
    def get_config_type(self) -> Type[SNPretrainedConfig]:
        return SNMistralConfig


class MistralModelLoaderPlugin(ModelLoaderPlugin):
    def get_automodel_map(self) -> Dict[Type[PreTrainedModel], _BaseAutoModelClass]:
        return {
            SNMistralModel: AutoModel,
            SNMistralForCausalLM: AutoModelForCausalLM,
        }

    def get_config_type(self) -> Type[SNPretrainedConfig]:
        return SNMistralConfig


class MistralRuntime(CachedInferenceRuntime, model=SNMistralForCausalLM):
    pass
