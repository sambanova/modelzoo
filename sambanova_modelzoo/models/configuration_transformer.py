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
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

from sambanova_modelzoo.models.config import SNPretrainedConfig, apply_hf_config_overrides
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class ConstructorSubConfigFieldMismatchError(Exception):
    """This is an error that indicates that there is a mismatch between the sub-config fields in the source and
    target config classes' constructors. This is a non-recoverable error that signifies the implementation of
    the SambaNoba config has a constructor that does not match the sub-config fields in the PretrainedConfig
    it is extending.
    """


# ====================================
# ConfigurationTransformerPlugin
# ====================================
class ConfigurationTransformerPlugin(abc.ABC):
    """
    A configuration transformation plugin based system.
    Define a plugin for each model to allow conversion from Hugging Face transformer
    config to a SambaNova config.
    Plugins are all subclasses of ConfigurationTransformerPlugin, and they are automatically
    added to the plugin list when imported, once.
    """
    # This is needed due to creation inside __init_subclass__ does not respect abstract method,
    # blame Guido for not fixing it: https://bugs.python.org/issue35815 (last comment)
    _registered_plugins_classes: List[Type['ConfigurationTransformerPlugin']] = list()
    _registered_plugins: Dict[Type['ConfigurationTransformerPlugin'], 'ConfigurationTransformerPlugin'] = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # This is needed due to creation inside __init_subclass__ does not respect abstract method,
        # blame Guido for not fixing it: https://bugs.python.org/issue35815 (last comment)
        ConfigurationTransformerPlugin._registered_plugins_classes.append(cls)

        plugin = cls()
        to_config = plugin.get_target_conversion_type()
        model_type_id = to_config.get_model_type_id()

        transform_map = plugin.get_architectures_transform_map()
        if len(transform_map) != len(set(transform_map.values())):
            raise ValueError(f"The get_architectures_transform_map for model type '{model_type_id}' is not bijective.")

        from transformers import AutoConfig
        AutoConfig.register(model_type_id, to_config, exist_ok=False)  # add the default exist_ok=False, explicitly

    @abstractmethod
    def get_source_conversion_type(self) -> Type[PretrainedConfig]:
        """
        Returns: The type of subclass the plugin converts from.
        """
    @abstractmethod
    def get_target_conversion_type(self) -> Type[SNPretrainedConfig]:
        """
        Returns: The type of subclass the plugin converts to.
        """
    @abstractmethod
    def get_architectures_transform_map(self) -> Dict[Type[PreTrainedModel], Type[PreTrainedModel]]:
        """
        Returns: Map of model architectures from hf to sn, e.g. MistralForCausalLM to SNMistralForCausalLM.
        """
    def is_match(self, source_config: Union[PretrainedConfig, SNPretrainedConfig]) -> bool:
        """
        Args:
            source_config
        Returns:
            True if source_config matches the plugin's source conversion type or itself (self conversion is ok)
        """
        # Make sure not to match subclasses, else two different subclasses of e.g. a config will match the same plugin
        source_type = type(source_config)
        return source_type is self.get_source_conversion_type() or source_type is self.get_target_conversion_type()

    def apply(self, config: Union[PretrainedConfig, SNPretrainedConfig],
              sn_model_args: Dict[str, Any]) -> Optional[Union[PretrainedConfig, SNPretrainedConfig]]:
        """
        Args:
            config: the config object to transform into a new SambaNova model config. May be target type already
                    self-transformation is supported.
            sn_model_args: the args we should choose from when constructing the new SNPretrainedConfig subclass object.
                           This may safely be a superset containing parameters that the specific SN config will not use.

        Returns:
            A new specific SambaNova config (that is also a PretrainedConfig as well as a SNPretrainedConfig)
        """
        if not isinstance(config, self.get_source_conversion_type()):
            raise ValueError(f"Expected config to be of matched type {type(self.get_source_conversion_type())} "
                             f"but got {type(config)}")

        target_type = self.get_target_conversion_type()
        sn_model_specific_args: Dict[str, Any] = target_type.get_sn_args(sn_model_args)
        self._sanity_check_model_specific_args(specific_args=sn_model_specific_args,
                                               provided_args=sn_model_args,
                                               target_type_name=target_type.__name__)

        sn_config = target_type.create(sn_args=sn_model_specific_args, original_config=config)

        if not isinstance(config, SNPretrainedConfig):
            self._remap_architectures_inplace(sn_config)

        return sn_config

    def _remap_architectures_inplace(self, sn_config: SNPretrainedConfig):
        if sn_config.architectures is None:
            return

        architecture_map_name = {
            hf_model_key.__name__: sn_model_value.__name__
            for hf_model_key, sn_model_value in self.get_architectures_transform_map().items()
        }
        # TODO: when trying to convert a model with unknown architecture, it will crash... handle more properly
        sn_config.architectures = [architecture_map_name[architecture] for architecture in sn_config.architectures]

    @staticmethod
    def get_registered_plugins() -> List['ConfigurationTransformerPlugin']:
        # This is needed due to creation inside __init_subclass__ does not respect abstract method,
        # blame Guido for not fixing it: https://bugs.python.org/issue35815 (last comment)
        for plugin_class in ConfigurationTransformerPlugin._registered_plugins_classes:
            if plugin_class not in ConfigurationTransformerPlugin._registered_plugins:
                ConfigurationTransformerPlugin._registered_plugins[plugin_class] = plugin_class()

        plugins = list(ConfigurationTransformerPlugin._registered_plugins.values())
        return sorted(plugins, key=lambda e: e.__class__.__name__)

    def _sanity_check_model_specific_args(self, specific_args: Dict[str, Any], provided_args: Dict[str, Any],
                                          target_type_name: str):
        if len(specific_args) != len(provided_args):
            ignored = ", ".join(f"{k}: {v}" for k, v in provided_args.items() if k not in specific_args)
            message = (f"You are passing 'sn_model_args' that are not defined in {target_type_name} "
                       f"nor SNPretrainedConfig! "
                       f"These arguments will be ignored: {{ {ignored} }}. This is OK if it is intentional."
                       f"If your intention is that these are SambaNova model patch args, "
                       f"then add them to {target_type_name} if they are model specific, or to SNPretainedConfig if "
                       f"they are general parameters for SN patched models.")
            logger.warn(message)


# ====================================
# ConfigurationTransformer
# ====================================
class ConfigurationTransformer:
    """
    A configuration transformation plugin-based system for converting Hugging Face
    PretrainedConfig to SambaNova configs.
    """
    def __init__(self, plugins: List[ConfigurationTransformerPlugin] = None):
        """
        Args:
            plugins: Optional. A list of unique plugin instances.
                     If None then use the globally registered list of plugins,
                     else use the specified list of plugin instances.
        """
        super().__init__()

        if plugins is not None and len(plugins) > 0:
            if not all([isinstance(p, ConfigurationTransformerPlugin) for p in plugins]):
                raise ValueError("plugins must be instances of subclasses of ConfigurationTransformer.")
            self._plugins = plugins
        else:
            self._plugins = ConfigurationTransformerPlugin.get_registered_plugins()
            if len(self._plugins) == 0:
                raise ValueError(
                    "Expected non-empty plugins list. Nothing matches an empty plugins list. "
                    "For transformers-based models, check the transformers package version and see if "
                    "it is compatible with the requirements.py. If incompatible, plugins will not be registered.")

    def run(self,
            config: PretrainedConfig,
            sn_model_args: Optional[Dict[str, Any]] = None,
            original_config_overrides: Optional[Dict[str, Any]] = None) -> Union[PretrainedConfig, SNPretrainedConfig]:
        """
        Applies exactly one plugin matching config.
        If not exactly one plugin matches, then we get an error, the behavior is undefined.
        Args:
            config: the Hugging Face config to transform into a SambaNova config
            sn_model_args: the args to choose from when constructing the new SNPretrainedConfig subclass object.
            original_config_overrides: additional overrides meant for overriding the original 'config'.

        Returns:
            A transformed config or None if no transformation plugin was applied.

        Raises:
            ValueError if not exactly 1 plugin matches
            ConstructorSubConfigFieldMismatchError if SN config constructor does not match the HF config
            constructor sub-config fields
        """
        overridden_hf_config = apply_hf_config_overrides(config=config, overrides=original_config_overrides or {})
        return self._run_nested(overridden_hf_config, sn_model_args)

    def _run_nested(self, config: PretrainedConfig,
                    sn_model_args: Optional[Dict[str, Any]] = None) -> Union[PretrainedConfig, SNPretrainedConfig]:

        sn_overrides = deepcopy(sn_model_args) if sn_model_args is not None else {}
        plugin = self.get_plugin_for_source_config(config)

        # Find sub configs in the PretrainedConfig instance,
        #  based on field names in the SNPretrainedConfig instance.
        target_config_type = plugin.get_target_conversion_type()
        source_config_type = config.__class__  # since we allow self-conversion

        sn_sub_configs = target_config_type.get_sn_subclass_nested_config_args()

        # Can't rely on __init__ signature for HF configs
        hf_sub_configs = {name: clazz for name, clazz in vars(config).items() if isinstance(clazz, PretrainedConfig)}

        # Checking types of sub-configs is tricky, since they don't strictly match... SNLlamaConfig <-> LlamaConfig
        #  We will settle for checking field names, we could use the plugins but that is a bit convoluted.
        #  We can allow hf to have sub_configs and SN to not have it (maybe hard coded in __init__)
        #  but if SN has sub_configs there must be at least that many also in the HF config.
        if set(sn_sub_configs.keys()) != set(hf_sub_configs.keys()):  # must match on sub-config fields
            raise ConstructorSubConfigFieldMismatchError(
                f'Expected class {target_config_type} to have matching sub-config fields '
                f'with {source_config_type}, in the constructor. '
                f'SN: {[f"n:{c}" for c in sn_sub_configs.keys()]}, '
                f'HF: {[f"n:{c}" for c in hf_sub_configs.keys()]}')

        collected_sub_configs = {}

        # Recurse if sub-configs
        for field, config_type in sn_sub_configs.items():
            sub_config_to_convert = getattr(config, field)
            sub_sn_overrides = sn_overrides.get(field, {})
            sn_overrides.pop(field, None)

            converted_config = self._run_nested(config=sub_config_to_convert, sn_model_args=sub_sn_overrides)
            collected_sub_configs[field] = converted_config

        # Base case, no sub-configs
        #  Use the sn_model_args to set the sub-config fields.
        #  This implies that the SN config must match the hugging face
        #  config structure for the sub-config fields.
        sn_model_args_with_sub_configs = {**collected_sub_configs, **sn_overrides}
        config = plugin.apply(config, sn_model_args=sn_model_args_with_sub_configs)
        return config

    def get_plugins(self) -> List[ConfigurationTransformerPlugin]:
        return self._plugins.copy()

    def get_plugin_for_source_config(self, config: PretrainedConfig) -> ConfigurationTransformerPlugin:
        """
        Get the matching plugin for config.
        If not exactly one plugin matches, then we get an error, the behavior is undefined.
        Args:
            config: the Hugging Face config to transform into a SambaNova config, or any subclass of PretrainedConfig
                    including specific sn configs.

        Returns:
            The plugin that matches the provided config.

        Raises:
            ValueError if not exactly 1 plugin matches
        """
        # We may want to allow multiple mappings, but in that case we need to add an explicit choice of mapping
        matching_plugins = [p for p in self._plugins if p.is_match(config)]
        if len(matching_plugins) != 1:
            raise ValueError(f"Misconfiguration in plugins, exactly 1 plugin should match! Got "
                             f"{len(matching_plugins)} matches for the config of type: "
                             f"{type(config).__name__}. Active plugins: "
                             f"{[p.__class__.__name__ for p in self._plugins]}")
        return matching_plugins[0]

    def get_plugin_for_type_id(self, model_type_id: str) -> ConfigurationTransformerPlugin:
        """
        Get the matching plugin for model_type_id as defined in the corresponding SambaNova config (target config)
        If not exactly one plugin matches, then we get an error, the behavior is undefined.
        Args:
            model_type_id: a string that identifies the model type.

        Returns:
            The plugin for the given model_type_id
        """
        # We may want to allow multiple mappings, but in that case we need to add an explicit choice of mapping
        matching_plugins = [
            p for p in self._plugins if model_type_id == p.get_target_conversion_type().get_model_type_id()
        ]
        if len(matching_plugins) != 1:
            raise ValueError(f"Misconfiguration in plugins, exactly 1 plugin should match! "
                             f"Got {len(matching_plugins)} matches for the model_type_id: {model_type_id}. "
                             f"Active plugins: {[p.__class__.__name__ for p in self._plugins]}")

        return matching_plugins[0]

    def is_model_type_id_registered(self, model_type_id: str) -> bool:
        """
        Args:
            model_type_id: a string, that identifies a SambaNova model. Defined in the SambaNova configuration for that model.

        Returns:
            True if a plugin has been registered that matches the model_type_id, else False
        """
        return any(True for p in self._plugins if model_type_id == p.get_target_conversion_type().get_model_type_id())
