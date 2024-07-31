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

import abc
import inspect
import json
import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, ConfigDict, Field
from sambanova_modelzoo.models.config import SNPretrainedConfig


class WhitelistConfigSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sn_config: Dict[str, Any] = Field(..., description="The model configuration for this whitelist config")
    job_configs: Dict[str, List[Dict[str, Any]]] = Field(
        ...,
        description=
        "A dictionary with each entry maps a job config's class name to a list containing key-value pairs derived from the attributes of that class."
    )


def generate_whitelist_config(model_config: SNPretrainedConfig,
                              job_config: Union['RDUGenerationAppConfig', 'RDUTrainingConfig'],
                              output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    A utility function for generating a whitelist configuration and optionally output the configuration at the specified output path.

    Args:
        model_config: The model configuration
        job_config: The job configuration
        output_path: Path of output file
    Returns: 
        the whitelist config. 
    """
    model_validator = SNAutoConfigValidator.get_model_validator(model_config)
    job_config_keys = model_validator.get_job_config_whitelist_keys()
    model_config_keys = model_validator.get_model_config_whitelist_keys()

    def filter_nested_dict(nested_dict: Dict[str, Any], keys: Set[str]):
        """
        Output a new dictionary containing only the keys in `keys`.
        """
        name_keys = {}
        for key, value in nested_dict.items():
            if key in keys:
                name_keys[key] = value
            elif isinstance(value, dict):
                result = filter_nested_dict(value, keys)
                if result is not None:
                    name_keys[key] = result
        return name_keys if name_keys else None

    job_config_cls_name = type(job_config).__name__
    job_whitelist_config = filter_nested_dict(job_config.model_dump(), job_config_keys)
    # remove redundant batchsize field in samba_compile
    if 'samba_compile' in job_whitelist_config and 'batch_size' in job_whitelist_config['samba_compile']:
        del job_whitelist_config['samba_compile']['batch_size']
    whitelist_config = WhitelistConfigSchema(sn_config=filter_nested_dict(model_config.to_dict(), model_config_keys),
                                             job_configs={job_config_cls_name: [job_whitelist_config]})
    whitelist = [whitelist_config.model_dump()]
    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(whitelist, f, indent=4)
    return whitelist


def add_whitelist_config(config: Dict[str, Any], whitelist: List[Dict[str, Any]]):
    """
    A utility function that adds a configuration to a whitelist.

    Args:
        config: The configuration to add 
        whitelist: The whitelist
    """
    def unique_values(data: List[Dict[str, Any]]):
        """
        Returns only unique values. This is needed because set( List of Dicts) doesn't work
        Arguments:
            data: A list of values
        Returns:
            A list of unique values
        """
        unique_data = []
        for d in data:
            if d not in unique_data:
                unique_data.append(d)
        return unique_data

    def sanitize_config(config: Dict[str, Any]):
        """
        Sanitize config before adding to the whitelist
        Arguments:
            config: The configuration to sanitize
        Returns:
            santinized configuration. Raise assertion if config can not be sanitized
        """
        config = WhitelistConfigSchema(**config)
        for cls_name, cls_values in config.job_configs.items():
            config.job_configs[cls_name] = unique_values(cls_values)
        return config.model_dump()

    sanitized_config = sanitize_config(config)
    if len(whitelist) == 0:
        return [sanitized_config]
    found = False
    for whitelist_config in whitelist:
        # If config's model configuration is in the whitelist, merge config's job_configs to the corresponding whitelist's job_configs.
        if sanitized_config['sn_config'] == whitelist_config['sn_config']:
            for cls_name, cls_values in sanitized_config["job_configs"].items():
                combined_values = unique_values(whitelist_config["job_configs"].get(cls_name, []) + cls_values)
                whitelist_config["job_configs"][cls_name] = combined_values
            found = True
            break
    # If config's model configuration is not found in the whitelist, it indicates that this is a new model configuration; therefore, it should be appended to the whitelist.
    if not found:
        whitelist.append(sanitized_config)
    return whitelist


def merge_whitelist_config(whitelist1: List[Dict[str, Any]],
                           whitelist2: List[Dict[str, Any]],
                           output_path: Optional[str] = None):
    """
    A utility function that combines two whitelist configurations, 
    returns the merged whitelist and optionally outputs the merged whitelist
    at the specified output path.

    Args:
        whitelist1: The first whitelist 
        whitelist2: The second whitelist
        output_path: Path of output file
    Returns:
        the merged whitelist
    """
    final_whitelist = []
    for config in whitelist1 + whitelist2:
        final_whitelist = add_whitelist_config(config, whitelist=final_whitelist)
    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(final_whitelist, f, indent=4)
    return final_whitelist


class SNConfigValidatorPlugin(abc.ABC):
    """
    A configuration validator plugin based system.
    Define a plugin for each model to allow blacklist/whitelist configurations.
    Plugins are all subclasses of SNConfigValidatorPlugin, and they are automatically
    added to the plugin list when imported, once.
    """
    _registered_plugins: Dict[Type['SNConfigValidatorPlugin'], 'SNConfigValidatorPlugin'] = dict()
    _config_to_validator_mapping: Dict[Type['SNPretrainedConfig'], 'SNConfigValidatorPlugin'] = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SNConfigValidatorPlugin._registered_plugins[cls] = None

    def __init__(self) -> None:
        super().__init__()
        self.whitelist = None

    @abstractmethod
    def get_config_type(self) -> Type['SNPretrainedConfig']:
        """
        Returns: The type of SNPretrainedConfig subclass the plugin is for.
        """
    def get_job_config_whitelist_keys(self) -> Set[str]:
        """
        Returns: A set of keys from the job configuration. These keys will be used to generate a job_config whitelist and validate an input job_config. These keys can be overwritten by the child plugin
        """
        return set([
            "batch_size", "arch", "num_tiles", "run_early_tp", "n_chips", "tensor_parallel", "fp32_ln", "fp32_logits",
            "fp32_skip_add", "mixedp_attn", "max_seq_length", "compiler_mode", "optim_level",
            "enable_mixed_precision_ops", "enable_multi_input_broadcast", "inference", "o1_experimental_opts",
            "tiling_accum", "use_air_ce", "num_tiles", "all_regions_hbm", "use_o1_default_rules"
        ])

    def get_model_config_whitelist_keys(self) -> Set[str]:
        """
        Returns: A set of keys from the model configuration. These keys will be used to generate a model_config whitelist and validate an input model_config. These keys can be overwritten by the child plugin.
        """
        return set([
            "architectures", "fp32_ln", "fp32_logits", "fp32_skip_add", "hidden_act", "hidden_size",
            "intermediate_size", "max_position_embeddings", "mixedp_attn", "model_type", "num_attention_heads",
            "num_hidden_layers", "num_key_value_heads", "vocab_size", "use_segmented_softmax_attn"
        ])

    def _whitelist_path(self) -> str:
        """
        Returns: A whitelist file path.
        """
        whitelist_dir = os.path.dirname(inspect.getfile(self.__class__))
        return os.path.join(whitelist_dir, "configs", 'whitelist.json')

    def _get_whitelist(self) -> List[Dict[str, Any]]:
        """
        Returns: A list of allowed config dictionary.
        """
        if self.whitelist is None:
            self.whitelist = []
            whitelist_path = self._whitelist_path()
            if os.path.exists(whitelist_path):
                with open(whitelist_path, 'r') as f:
                    self.whitelist.extend(json.load(f))
        return self.whitelist

    def _is_whitelist_key_values_match(self, whitelist_config, job_config_dict: Dict[str, Any]):
        """
        Check if the whitelist's keys and values exist and match in job's configurations.
        
        Returns:
            True if all keys and values present in whitelist_config exist and are equal to those in job_config_dict,
            False otherwise.
        """
        # If whitelist_config is not a dictionary, compare values directly
        if not isinstance(whitelist_config, dict):
            return whitelist_config == job_config_dict
        # Iterate over keys and values in whitelist_config
        for key, value in whitelist_config.items():
            if key not in job_config_dict:
                return False
            # `arch` needs a differnt way to compare
            if key == "arch":
                job_value = job_config_dict[key]
                # Legacy `arch` field is a list of strings. If using legacy `arch` field, then take the first element.
                job_value = job_value[0] if isinstance(job_value, list) else job_value
                # Continue if arch='native' from job_config because we don't know the actual arch
                # Returns False if job_config and whitelist_config have different arch values
                if job_value != 'native' and job_value != whitelist_config[key]:
                    return False
                continue
            # Check if key exists in job_config_dict and recursively compare values
            if not self._is_whitelist_key_values_match(value, job_config_dict[key]):
                return False
        # All keys and values in whitelist_config are equal to those in job_config_dict
        return True

    def _is_supported_job_config(self, whitelist_config_dict,
                                 job_config: Union['RDUGenerationAppConfig', 'RDUTrainingConfig']) -> bool:
        """
        Determine if the job config is supported by the whitelist configuration.
        Args:
            whitelist_config_dict: A configuration dictionary from the whitelist
            job_config: The job configuration
        Returns:
            True if the job config is supported, False otherwise. 
        """
        job_config_dict = job_config.model_dump()
        supported_job_configs = whitelist_config_dict.get("job_configs", {})
        cls_name = type(job_config).__name__
        if cls_name not in supported_job_configs:
            return False
        for config in supported_job_configs[cls_name]:
            if self._is_whitelist_key_values_match(config, job_config_dict):
                return True
        return False

    def _is_supported_model_config(self, whitelist_config_dict: Dict[str, Any],
                                   model_config: SNPretrainedConfig) -> bool:
        """
        Determine if the model configuration is supported by the whitelist configuration.
        Args:
            model_config: The model configuration
            whitelist_config_dict: A configuration dictionary from the whitelist
        Returns:
            True if the model configuration is supported, False otherwise. 
        """
        model_config_dict = model_config.to_dict()
        supported_model_config = whitelist_config_dict["sn_config"]
        # If no model config is supported for this model or if no job configuration was passed, returns False
        if len(supported_model_config) == 0 or len(model_config_dict) == 0:
            return False
        return self._is_whitelist_key_values_match(supported_model_config, model_config_dict)

    def _is_supported_config(self, model_config: SNPretrainedConfig,
                             job_config: Union['RDUGenerationAppConfig', 'RDUTrainingConfig'],
                             whitelist_config_dict: Dict[str, Any]):
        """
        Determine if the current job configuration is supported by the whitelist configuration.
        Args:
            model_config: The model configuration
            job_config: The job configuration
            whitelist_config_dict: A whitelist configuration

        Returns:
            True if the config is supported, False otherwise. 
        """
        if not self._is_supported_model_config(whitelist_config_dict, model_config):
            return False
        if not self._is_supported_job_config(whitelist_config_dict, job_config):
            return False
        return True

    def validate(self, model_config: SNPretrainedConfig,
                 job_config: Union['RDUGenerationAppConfig', 'RDUTrainingConfig']):
        """
        Validate that model_config and job_config exist in whitelist.  Raise an error if they do not.
        Args:
            model_config: The model configuration
            job_config: The job configuration
        """
        for whitelist_config in self._get_whitelist():
            # If found a supported config, just return
            if self._is_supported_config(model_config, job_config, whitelist_config):
                return
        # If no supported config was found, raise an error
        raise ValueError(
            f"Configuration is not supported. Please see {self._whitelist_path()} for supported configurations. Or add `validate_config=False` to proceed with execution, but be aware that it might lead to program failure."
        )

    @staticmethod
    def get_model_config_plugin_mapping() -> Dict[Type['SNPretrainedConfig'], 'SNConfigValidatorPlugin']:
        """
        Get mapping from SNPretrainedConfig subclass to its corresponding SNConfigValidatorPlugin instance
        """
        for plugin_class, plugin_instance in SNConfigValidatorPlugin._registered_plugins.items():
            if plugin_instance is None:
                plugin_instance = plugin_class()
                SNConfigValidatorPlugin._registered_plugins[plugin_class] = plugin_instance
                SNConfigValidatorPlugin._config_to_validator_mapping[
                    plugin_instance.get_config_type()] = plugin_instance
        return SNConfigValidatorPlugin._config_to_validator_mapping.copy()


class SNAutoConfigValidator():
    """
    A configuration validator plugin based system for validating SambaNova configs.
    """
    @staticmethod
    def get_model_validator(model_config: SNPretrainedConfig) -> 'SNConfigValidatorPlugin':
        """
        Get a model validator based on sn model config. Raise an error if validator not found.
        Args:
            model_config: The model configuration
        """
        mapping = SNConfigValidatorPlugin.get_model_config_plugin_mapping()
        config_class = type(model_config)
        if config_class in mapping:
            return mapping[config_class]
        raise ValueError(
            f"Configuration is not supported. No validator found! Add `validate_config=False` to proceed with execution, but be aware that it might lead to program failure."
        )

    @staticmethod
    def validate(model_config: SNPretrainedConfig,
                 job_config: Union['RDUGenerationAppConfig', 'RDUTrainingConfig']) -> bool:
        """
        If `job_config.validate_config=True`, validate that `model_config` and `job_config` exist in the corresponding whitelist, raise an error if they do not.
        Args:
            model_config: The model configuration
            job_config: The job configuration

        Returns:
            None
        """
        # If users don't want to validate the job configuration or this is not a compilation job
        if not job_config.validate_config or job_config.command != "compile":
            return
        # Only generate whitelist during compilation
        if job_config.generate_whitelist_config:
            # (TODO): get the actual output folder from samba directly, instead of this hack
            whitelist_output_path = os.path.join(job_config.samba_compile.output_folder,
                                                 job_config.samba_compile.pef_name, "whitelist.json")
            generate_whitelist_config(model_config=model_config,
                                      job_config=job_config,
                                      output_path=whitelist_output_path)
        model_validator = SNAutoConfigValidator.get_model_validator(model_config)
        model_validator.validate(model_config, job_config)
