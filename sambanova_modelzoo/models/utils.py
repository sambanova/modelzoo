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

import sys
from typing import Any, Dict, Optional

import torch
from sambanova_modelzoo.libs.common.pretrained_model_schema import PretrainedModelConfig
from sambanova_modelzoo.models.config import SNPretrainedConfig
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformer
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


def is_jit() -> bool:
    """Determines if code being run is through JIT or Samba by checking if torch_rdu is imported"""
    return 'torch_rdu' in sys.modules


def init_weights(self, module: torch.nn.Module):
    """Model Zoo init weight function selector between JIT and Samba"""
    if is_jit():
        # [TODO] Investigate using LazyModules for init_weights
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    else:
        from sambanova_modelzoo.models.lazy_init import sn_init_weights
        sn_init_weights(self, module)


def logger_info(message: str):
    """Model Zoo logging function that only prints for Samba"""
    if not is_jit():
        from sambaflow.logging import samba_logger
        samba_logger.info(message)


def get_sn_config(config_name_or_path: str,
                  sn_model_config: PretrainedModelConfig,
                  original_config_overrides: Optional[Dict[str, Any]] = None,
                  validate_config: bool = False) -> SNPretrainedConfig:
    """
    Use original Hugging Face model checkpoint to load a SambaNova model config.
    SambaNova model config is extended from Hugging Face for RDU optimization. 
    Args:
        config_name_or_path: Either Hugging Face model config file or directory that contains the config file or model id for downloading.
        sn_model_config: SambaNova specific pretrained model configurations.
        original_config_overrides: Configs that override the original model configurations.
        validate_config: Check if the model config is officially supported.
    Returns:
        SambaNova pretrained model config for constructing a model.
    """

    # Load the model's Hugging Face config.
    # This can be from a config.json file, checkpoint folder or model identifier.
    config = AutoConfig.from_pretrained(config_name_or_path)

    # Embed specific model instantiation details using ConfigurationTransformer().
    # Consult config/schema.py for a complete list of parameters.
    sn_pretrained_config = sn_model_config.model_dump()
    sn_config = ConfigurationTransformer().run(config,
                                               sn_model_args=sn_pretrained_config,
                                               original_config_overrides=original_config_overrides)
    return sn_config


def load_model_from_config(config_name: str,
                           sn_model_config: PretrainedModelConfig,
                           original_config_overrides: Optional[Dict[str, Any]] = None,
                           validate_config: bool = False) -> PreTrainedModel:
    """
    Use original Hugging Face model checkpoint to load a SambaNova model.
    SambaNova model is modified from Hugging Face for RDU optimization. 
    Args:
        config_name: Either Hugging Face model config file or directory contains the config file or model id for downloading.
        sn_model_config: SambaNova specific pretrained model pydantic configuration schema.
        original_config_overrides: Configs that override the original model configurations.
        validate_config: Check if the model config has been officially supported.
    Returns:
        SambaNova customized model.
    """
    sn_config = get_sn_config(config_name, sn_model_config, original_config_overrides=original_config_overrides)
    # dtype has to be bfloat16 to do mixed precision on RDU
    sn_model = AutoModelForCausalLM.from_config(sn_config, torch_dtype=sn_model_config.dtype)
    return sn_model


def load_model_from_pretrained(model_name_or_path: str,
                               sn_model_config: PretrainedModelConfig,
                               original_config_overrides: Optional[Dict[str, Any]] = None,
                               validate_config: bool = False) -> PreTrainedModel:
    """
    Use original Hugging Face model checkpoint to load a SambaNova model.
    SambaNova model is modified from Hugging Face for RDU optimization. 
    Args:
        model_name_or_path: Has either Hugging Face model config file or model cache directory or model id for downloading.
        sn_model_config: SambaNova specific pretrained model pydantic configuration schema.
        original_config_overrides: Configs that override the original model configurations.
        validate_config: Check if the model config has been officially supported.
    Returns:
        SambaNova customized model.
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    # If original config has weight tied and user overrides it to be untied, it is interpreted as force_untie
    force_untie = original_config_overrides and \
            not original_config_overrides.get('tie_word_embeddings', True) and \
            config.tie_word_embeddings

    sn_config = get_sn_config(model_name_or_path, sn_model_config, original_config_overrides=original_config_overrides)
    # dtype has to be bfloat16 to do mixed precision on RDU

    # Tie embeddings to load the lm_head weights correctly
    if force_untie:
        sn_config.tie_word_embeddings = True
    sn_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    config=sn_config,
                                                    torch_dtype=sn_model_config.dtype)

    # Force to untie the weights after loading the checkpoints by cloning. Sambaflow stack does not support multigraph
    # cached inference with weights tied. So here, we use the similar way of cloning the tied weights as Hugging Face
    # did for torchscript, ref transformers/modeling_utils.py::tie_or_clone_weights
    if force_untie:
        sn_model.get_output_embeddings().weight = torch.nn.Parameter(sn_model.get_input_embeddings().weight.clone())
        # After weights being loaded and untied through cloning, change the model config to match the weight
        sn_model.config.tie_word_embeddings = False
    return sn_model
