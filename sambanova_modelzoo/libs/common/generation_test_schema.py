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

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, PositiveInt, model_validator
from sambanova_modelzoo.libs.common.pretrained_model_schema import PretrainedModelConfig
from sambanova_modelzoo.libs.common.samba_schema import SambaConfig, ValidatorConfig
from typing_extensions import Self


class GenerationConfig(BaseModel):
    """ Config for text generation including testing purposed flags """
    model_config = ConfigDict(extra='forbid')

    batch_size: PositiveInt = Field(description="Static batch size", default=1)
    fuse_lm_head_with_postprocess: bool = Field(
        description="Fuse lm_head with Postprocess graph for better performance", default=True)
    prompts: List[str] = Field(description="Prompt input")
    max_new_tokens: Optional[PositiveInt] = Field(description="Maximum generated tokens", default=None)
    expected_completions: Optional[List[str]] = Field(description="Expected completion sentences", default=None)
    profile: bool = Field(description="Whether to do profiling", default=False)
    seed: PositiveInt = Field(description="Random seed on torch/numpy/python for reproducibility", default=12345)
    test_out_of_bound: bool = Field(description="Testing out-of-bound generation on longer inputs", default=False)
    generation_strategy: Optional[Dict[str, Any]] = Field(
        description="Additional parameters to huggingface generation call", default=None)
    output_top_logits: bool = Field(description="If true, output a table of top 3 logits per token per prompt.",
                                    default=False)
    static_seq_lengths: Optional[List[PositiveInt]] =\
            Field(description="Static sequence length to pad the input_ids for cache generation graph", default=None)

    model_name_or_path: Union[str, FilePath, DirectoryPath] = Field(
        description=("Model identifier on Hugging Face to download on the fly or path to model config json file "
                     "or directory to downloaded Hugging Face checkpoint cache"), )

    @model_validator(mode='after')
    def prompts_completion(self) -> Self:
        if self.expected_completions is not None and len(self.prompts) != len(self.expected_completions):
            raise ValueError(
                f"Number of prompts {len(self.prompts)} has to match number of expected completions {len(self.expected_completions)}"
            )
        return self


class RDUGenerationAppConfig(SambaConfig, ValidatorConfig):
    """ Config for text generation application including testing purposed flags """
    model_config = ConfigDict(extra='forbid')

    model: PretrainedModelConfig = Field('Model specific arguments')
    generation: GenerationConfig = Field('Application specific arguments')

    @model_validator(mode='after')
    def prompts(self) -> Self:
        """ Trim prompts and completions to batch_size if batch_size < prompts """
        if self.command == 'run':
            if self.generation.batch_size > len(self.generation.prompts):
                raise ValueError(
                    f"Number of prompt samples {len(self.generation.prompts)} cannot be smaller than batch size {self.generation.batch_size}"
                )
            else:
                self.generation.prompts = self.generation.prompts[:self.generation.batch_size]
                self.generation.expected_completions = self.generation.expected_completions[:self.generation.batch_size]
        return self

    @model_validator(mode='after')
    def validate_static_seq_lengths(self) -> Self:
        """ Generate defaults using max_seq_length """
        return _validate_static_seq_lengths(self)

    @model_validator(mode='after')
    def use_plugin_heuristics(self) -> Self:
        """ Ensure the o1hd flag is consistent between model config and samba_compile config """
        if self.command == 'compile':
            use_o1hd = self.model.use_plugin_heuristics or self.samba_compile.use_plugin_heuristics
            self.model.use_plugin_heuristics = use_o1hd
            self.samba_compile.use_plugin_heuristics = use_o1hd
        return self

    @model_validator(mode='after')
    def profile(self) -> Self:
        """ Append compiler verbose and debug options if profile is True """
        if self.command == 'compile':
            if self.generation.profile == True:
                self.samba_compile.verbose = 1
                self.samba_compile.debug = True
        return self


class CPUGenerationAppConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: PretrainedModelConfig
    generation: GenerationConfig

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
