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

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, model_validator


class CheckpointConfig(BaseModel, extra='forbid'):
    """ Parameters controlling loading a specific checkpoint (common to CPU and RDU) """
    # Tell pydantic it's okay to have fields starting with model_
    # model_config is used by pydantic (Base "Model"), not the app
    model_config = ConfigDict(protected_namespaces=('protected_', ), extra='forbid')

    config_name: Optional[Union[str, FilePath]] = Field(
        description=("Model identifier on Hugging Face to download on the fly or "
                     "path to model config json file or "
                     "directory that contains the config json file"),
        default=None)
    # Changing the checkpoint does not require recompiling a pef; unless the model architecture is different
    model_name_or_path: Optional[Union[str, DirectoryPath]] = Field(
        description=("Model identifier on Hugging Face to download on the fly or "
                     "directory to downloaded Hugging Face checkpoint cache"),
        default=None)

    @model_validator(mode="after")
    def validate_config_or_cache(self):
        """ Check that user provided exactly one of config_name or model_name_or_path """
        if bool(self.config_name) == bool(self.model_name_or_path):
            raise ValueError('Please specify exactly one of checkpoint.config_name or checkpoint.model_name_or_path')
        return self
