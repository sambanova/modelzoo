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
from abc import abstractmethod
from typing import Dict, List, Type

from sambanova_modelzoo.models.config import SNPretrainedConfig
from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass


# ====================================
# ModelLoaderPlugin
# ====================================
class ModelLoaderPlugin(abc.ABC):
    """
    A model loader plugin based system.
    Define a plugin for each model to allow for AutoModel registrations.
    Plugins are all subclasses of ModelLoaderPlugin, and they are automatically
    added to the plugin list when imported, once.
    """
    _registered_plugins_classes: List[Type['ModelLoaderPlugin']] = list()
    _registered_plugins: Dict[Type['ModelLoaderPlugin'], 'ModelLoaderPlugin'] = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelLoaderPlugin._registered_plugins_classes.append(cls)
        plugin = cls()
        cls._register_models(loaders=plugin.get_automodel_map(), config_class=plugin.get_config_type())

    @classmethod
    def _register_models(cls, loaders: Dict[Type[PreTrainedModel], _BaseAutoModelClass],
                         config_class: SNPretrainedConfig):
        """"
        Args:
            loaders: Model, AutoModel pair that will be registered.
            config_class: SNConfig for the models you will register with the AutoModels.
        """
        for model, auto_model in loaders.items():
            auto_model.register(config_class, model)

    @staticmethod
    def get_registered_plugins() -> List['ModelLoaderPlugin']:
        for plugin_class in ModelLoaderPlugin._registered_plugins_classes:
            if plugin_class not in ModelLoaderPlugin._registered_plugins:
                ModelLoaderPlugin._registered_plugins[plugin_class] = plugin_class()

        return list(ModelLoaderPlugin._registered_plugins.values())

    def is_match(self, config: SNPretrainedConfig) -> bool:
        """
        Args:
            config
            # TODO: Need more details
        Returns:
            True if the config matches the plugin's config type. 
            # TODO: For example? 
        """
        return type(config) is self.get_config_type()

    @abstractmethod
    def get_config_type(self) -> Type[SNPretrainedConfig]:
        """
        Returns: The type of config for the plugin.
        """
    @abstractmethod
    def get_automodel_map(self) -> Dict[Type[PreTrainedModel], _BaseAutoModelClass]:
        """
        Returns: Mapping of SambaNova model to the correct corresponding AutoModel, e.g. SNMistralForCausalLM to AutoCausalLLM.
        """


class ModelLoader:
    def __init__(self, plugins: List[ModelLoaderPlugin] = None):
        """
        Args:
            plugins: optional, if None, then use the registered list of plugins, else use the specified list.
        """
        super().__init__()
        self._plugins = plugins if plugins is not None else ModelLoaderPlugin.get_registered_plugins()

    def get_plugins(self) -> List[ModelLoaderPlugin]:
        return self._plugins.copy()
