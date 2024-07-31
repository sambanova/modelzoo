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

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class PretrainedModelConfig(BaseModel):
    """
    Parameters controlling modifications to the model architecture at initialization (common to CPU and RDU).
    This allows for additional control of the model that does not exists in base Huggingface config.json.
    These parameters are passed to ConfigurationTransformer().run() when loading a model from ModelZoo.

    This list is compiled from the following sources
        models/config.py - common parameters for all LLMs models
        models/<model_type>/configuration_<model_type>.py - additional parameters for specific models
    """

    model_config = ConfigDict(extra='forbid')

    fp32_ln: bool = Field(description="Layernorm fp32/mixp computation, see SNPretrainedConfig")
    fp32_logits: bool = Field(description="Logits fp32/mixp computation, see SNPretrainedConfig")
    fp32_skip_add: bool = Field(description="Residual add in fp32, see SNPretrainedConfig")
    mixedp_attn: bool = Field(description="Attention in mixp, see SNPretrainedConfig")
    max_seq_length: PositiveInt = Field(description="Maximum sequence length including prompt and new tokens generated")
    use_plugin_heuristics: bool = Field(description="Enable O1HD heuristics", default=False)
    use_segmented_softmax_attn: bool = Field(description="SDPA flash-attention, see SNPretrainedConfig", default=False)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """ Model weight's dtype, dependent on mixedp_attn """
        return torch.bfloat16 if self.mixedp_attn else None
