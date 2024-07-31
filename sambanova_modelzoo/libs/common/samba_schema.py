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

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sambaflow.samba.argconf.args import CompileArgs, RunArgs


class SambaConfig(BaseModel):
    """
    A convenient Pydantic model to build app that supports compile and run
    """
    model_config = ConfigDict(extra='forbid')
    command: Literal['compile', 'run']
    samba_compile: Optional[CompileArgs] = Field(default=None, description="Compile time arguments")
    samba_run: Optional[RunArgs] = Field(default=None, description="Runtime arguments")

    @model_validator(mode='before')
    @classmethod
    def conditional_parse(cls, data: Dict) -> Dict:
        """ Parse either compile or run command conditioned on command """
        if data['command'] == 'compile' and 'samba_run' in data:
            del data['samba_run']
        elif data['command'] == 'run' and 'samba_compile' in data:
            del data['samba_compile']
        return data


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    validate_config: bool = Field(
        description=
        "Specifies whether the program should continue execution without performing configuration validations for compilation, even though failure is possible.",
        default=True)
    generate_whitelist_config: bool = Field(
        description="Whether to output whitelist.json for the config in the output folder", default=False)
