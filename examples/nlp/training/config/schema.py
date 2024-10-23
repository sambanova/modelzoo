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

"""
This file contains a set of Pydantic models that define program arguments.

The most important thing to note is that RDU training requires
the same arguments as CPU training, plus arguments to compile a pef.
"""

from typing import Optional

from pydantic import BaseModel, DirectoryPath, Field, PositiveFloat, PositiveInt, model_validator
from sambanova_modelzoo.libs.common.common_schema import CheckpointConfig
from sambanova_modelzoo.libs.common.pretrained_model_schema import PretrainedModelConfig
from sambanova_modelzoo.libs.common.samba_schema import SambaConfig, ValidatorConfig
from typing_extensions import Self


class TrainingConfig(BaseModel, extra='forbid'):
    """ Parameters controlling the training run (common to CPU and RDU) """

    dataset: DirectoryPath = Field(description="Path to a dataset prepared using generative_data_prep")
    batch_size: Optional[PositiveInt] = Field(description="The (static) number of samples to train over at a time",
                                              requires_recompile=True)
    num_epochs: PositiveInt = Field(default=1, description="The number of times to iterate over the dataset")
    end_early_at_step: Optional[PositiveInt] = Field(default=None, description="The number of training steps to run. " \
            "Intended for testing only as this must be less than the length of the dataset")
    learning_rate: PositiveFloat = Field(default=1e-5, description="The starting learning rate for the optimizer")
    output_dir: str = Field(default='trained_model', description="Path to folder to save trained checkpoint")
    evaluate: bool = Field(default=False, description="Whether to evaluate the model on the dev set after each epoch")
    seed: int = Field(default=42, description='Seed parameter for random number generation')


class CPUTrainingConfig(BaseModel):
    model: PretrainedModelConfig
    checkpoint: CheckpointConfig
    training: TrainingConfig


class RDUTrainingConfig(CPUTrainingConfig, SambaConfig, ValidatorConfig):
    """
    Collection of Arguments for RDU training app

    The combined pydantic model consists of:
    (from sambanova_modelzoo.libs.common.samba_schema.SambaConfig)
        command:
        samba_compile:
        samba_run:
    (from CPUTrainingConfig)
        model:
        checkpoint:
        training:
    """
    @model_validator(mode="before")
    def no_require_dataset_for_compile(self) -> Self:
        """ Fill in a dummy argument for training.dataset while compiling since it's not used """
        if self['command'] == 'compile':
            self['training']['dataset'] = '.'
        return self

    @model_validator(mode="after")
    def no_weight_loading_during_compilation(self) -> Self:
        """ Compilation does not need to load weight data """
        if self.command == 'compile' and self.checkpoint.model_name_or_path:
            raise ValueError(f"Use checkpoint.config_name during compilation to skip weight loading")
        return self
