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
from sambanova_modelzoo.models.gemma.configuration_gemma import SNGemmaConfig
from sambanova_modelzoo.models.gemma.modeling_gemma import SNGemmaForCausalLM, SNGemmaModel
from sambanova_modelzoo.models.model_loader import ModelLoaderPlugin
from transformers import AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM


class GemmaConfigurationTransformer(ConfigurationTransformerPlugin):
    def get_source_conversion_type(self) -> Type[PretrainedConfig]:
        return GemmaConfig

    def get_target_conversion_type(self) -> Type[SNPretrainedConfig]:
        return SNGemmaConfig

    def get_architectures_transform_map(self) -> Dict[Type[PreTrainedModel], Type[PreTrainedModel]]:
        return {
            GemmaForCausalLM: SNGemmaForCausalLM,
        }


class GemmaConfigValidator(SNConfigValidatorPlugin):
    def get_config_type(self) -> Type[SNPretrainedConfig]:
        return SNGemmaConfig


class GemmaModelLoaderPlugin(ModelLoaderPlugin):
    def get_automodel_map(self) -> Dict[Type[PreTrainedModel], _BaseAutoModelClass]:
        return {SNGemmaModel: AutoModel, SNGemmaForCausalLM: AutoModelForCausalLM}

    def get_config_type(self) -> Type[SNPretrainedConfig]:
        return SNGemmaConfig


class GemmaRuntime(CachedInferenceRuntime, model=SNGemmaForCausalLM):
    pass
