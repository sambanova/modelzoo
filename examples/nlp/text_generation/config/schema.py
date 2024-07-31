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

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from sambanova_modelzoo.libs.common.common_schema import CheckpointConfig
from sambanova_modelzoo.libs.common.pretrained_model_schema import PretrainedModelConfig
from sambanova_modelzoo.libs.common.samba_schema import SambaConfig, ValidatorConfig
from typing_extensions import Self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompts: List[str] = Field(description="Prompt input")
    max_new_tokens: Optional[PositiveInt] = Field(description="Maximum generated tokens", default=None)
    profile: bool = Field(description="Whether to do profiling", default=False)
    seed: PositiveInt = Field(description="Random seed on torch/numpy/python for reproducibility", default=12345)
    batch_size: PositiveInt = Field(description="Static batch size")
    static_seq_lengths: Optional[List[PositiveInt]] =\
            Field(description="Static sequence length to pad the input_ids for cache generation graph", default=None)


class RDUGenerationAppConfig(SambaConfig, ValidatorConfig):
    model_config = ConfigDict(extra='forbid')
    model: PretrainedModelConfig = Field('Model specific arguments')
    generation: GenerationConfig = Field('Application specific arguments')
    checkpoint: CheckpointConfig = Field("Hugging Face checkpoint arguments")

    @model_validator(mode='after')
    def batch_size(self) -> Self:
        """ batch_size and prompts has to match at runtime """
        if self.command == 'run' and self.generation.batch_size != len(self.generation.prompts):
            raise ValueError(
                f"Number of prompt samples {len(self.generation.prompts)} does not match batch size {self.generation.batch_size}"
            )
        return self

    @model_validator(mode='after')
    def validate_static_seq_lengths(self) -> Self:
        """ Generate defaults using max_seq_length """
        return _validate_static_seq_lengths(self)


class CPUGenerationAppConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: PretrainedModelConfig = Field('Model specific arguments')
    generation: GenerationConfig = Field('Application specific arguments')
    checkpoint: CheckpointConfig = Field("Hugging Face checkpoint arguments")

    @model_validator(mode='after')
    def validate_static_seq_lengths(self) -> Self:
        """ Generate defaults using max_seq_length """
        return _validate_static_seq_lengths(self)


def _validate_static_seq_lengths(app_config: Union[RDUGenerationAppConfig, CPUGenerationAppConfig]) -> Self:
    """ Generate defaults using max_seq_length """
    if app_config.generation.static_seq_lengths is None:
        app_config.generation.static_seq_lengths = [app_config.model.max_seq_length]
    if any(s > app_config.model.max_seq_length for s in app_config.generation.static_seq_lengths):
        raise ValueError(
            f"static_seq_lengths {app_config.generation.static_seq_lengths} cannot be larger than max_seq_length {app_config.model.max_seq_length}"
        )
    return app_config
