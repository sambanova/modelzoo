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

import os
import tempfile
from accelerate import init_empty_weights
from filelock import FileLock
from transformers import AutoConfig, PreTrainedModel, AutoModelForCausalLM
from sambaflow import samba

from config.schema import CheckpointConfig

def save_as_huggingface_checkpoint(model: PreTrainedModel, checkpoint: CheckpointConfig, output_dir: str):
    """Save a traced samba model to a huggingface checkpoint"""

    # Copy tensor data from RDU to host
    samba.session.to_cpu(model)

    # Convert SambaTensors to Torch tensors
    torch_sd = {}
    for name, param in model.state_dict().items():
        torch_sd[name] = samba.to_torch(param)
    
    # Create an identical torch model as the base model
    model_pointer = checkpoint.config_name or checkpoint.model_name_or_path
    config = AutoConfig.from_pretrained(model_pointer)
    torch_model = AutoModelForCausalLM.from_config(config)
    torch_model.load_state_dict(torch_sd)

    # Save the model
    torch_model.save_pretrained(output_dir)
