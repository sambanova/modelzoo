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

from typing import Any, Dict, Optional

from sambanova_modelzoo.models.config import SNPretrainedConfig
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformer
from transformers import AutoConfig, PreTrainedTokenizer


def get_config_overrides_for_generation(pad_token_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Args:
        pad_token_id: Padding token id to overwrite the original model config.
    Returns:
        Text generation model config overrides for RDU.
    """
    original_config_overrides = {
        # Only cached inference is supported
        'use_cache': True,
        # Weights tying is not supported during cached inference compilation. They need to be traced as separate
        # symbols and duplicates in values. Please make sure to use or refer to models.utils's load_model_for_pretrained
        # to correctly load the checkpoints with weights tied at runtime.
        'tie_word_embeddings': False
    }
    if pad_token_id is not None:
        original_config_overrides['pad_token_id'] = pad_token_id
    return original_config_overrides


def get_sn_config_for_generation(model_name_or_path: str,
                                 pad_token_id: Optional[int] = None,
                                 original_config_overrides: Optional[Dict[str, Any]] = None,
                                 sn_model_args: Optional[Dict[str, Any]] = None) -> SNPretrainedConfig:
    """
    Get the model config from the cache directory and transform into SNPretrainedConfig for text generation.
    Args:
        model_name_or_path: Directory that contains the model config file.
        cache_dir: Original Hugging Face cache directory.
        pad_token_id: Padding token id to overwrite the original model config.
        original_config_overrides: Configs that overrides the original config in the model config file.
        sn_model_args: SambaNova specific SNPretrainedConfig.
        validate_config: Check if the model config has been officially supported.
    Returns:
        Transformed SNPretrainedConfig
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    hf_config = get_config_overrides_for_generation(pad_token_id=pad_token_id)
    if original_config_overrides:
        hf_config.update(original_config_overrides)

    sn_config = ConfigurationTransformer().run(config, sn_model_args=sn_model_args, original_config_overrides=hf_config)
    return sn_config


def configure_pad_token(tokenizer: PreTrainedTokenizer):
    """
    Update pad_token to eos_token if pad_token is not present in the tokenizer.
    Args:
        tokenizer: Transformers tokenizer
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
