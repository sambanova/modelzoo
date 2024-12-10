# Copyright 2023-2024 SambaNova Systems, Inc.
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

from sambanova_modelzoo.models.config import SNPretrainedConfig
from transformers.models.gemma.configuration_gemma import GemmaConfig


class SNGemmaConfig(GemmaConfig, SNPretrainedConfig):
    model_type = 'sngemma'

    def __init__(
            self,
            # ===== subclass specific args =====
            # ===== the args for hf and sn base classes
            **kwargs  # important! always have this.
    ):
        """
        This constructor takes the named args from both GemmaConfig and SNPretrainedConfig.
        All named arguments are split into arguments that are defined in SNPretrainedConfig and "the rest".
        These are subsequently passed to the SNPretrainedConfig constructor and the other inherited config's constructor.
        
        If you add a parameter to the constructor, it will not be passed to the superclasses, you must handle it in this constructor.
        See superclasses for default values.
        """
        SNPretrainedConfig.init_superclasses(subclass_self=self, kwargs_dict=kwargs)
