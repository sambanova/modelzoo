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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

from transformers import PretrainedConfig


class SNPretrainedConfig(ABC):
    model_type: str = ""

    def __init__(
            self,
            mixedp_attn: bool = False,
            fp32_logits: bool = False,
            fp32_ln: bool = False,
            fp32_skip_add: bool = False,
            lazy_init: bool = False,
            use_plugin_heuristics: bool = False,
            use_segmented_softmax_attn: bool = False,
            seg_softmax_block_size: Optional[int] = None,
            max_seq_length: Optional[int] = None,
    ):
        """
        SambaNova pretrained config
            Args:
                mixedp_attn: If true, use fp32/bf16 mixed precision attention for accurate training.
                fp32_logits: If true, cast logits to fp32 precision so the followup loss operator can use fp32 logits
                             input to operate in bf16/fp32 mixed precision mode.
                fp32_ln: If true, cast hidden states to fp32 so that layernorm is in fp32 precision
                fp32_skip_add: If true, enable fp32 residual connection for better precision.
                               For transformers type of architecture, examples of residual connections include:
                               1. input to (pre MHA layernorm -> MHA) adds to the output of MHA
                               2. input to (post MHA layernorm -> MLP) adds to the output of feedforward MLP
                lazy_init: Lazily initialize the weights for faster compilation and checkpoint loading.
                           It also uses the multithreaded random library to speed up initialization.
                           When used with Samba frontends, it will also stream the weight device transfer without
                           materializing the whole model.
                use_plugin_heuristics: Whether to use O1 HD heuristics for customized mapping
                use_segmented_softmax_attn: Use SDPA flashattention on RDU, to be deprecated by _attn_implementation
                                            after torch 2.0 upgrade
                seg_softmax_block_size: Force a key-value block size in segmented softmax attention (flash attention)
                                        in SDPA.
                max_seq_length: The sequence length of the KV cache used by token generation graph. It is the maximum
                                number of tokens one can generate including the input prompts.
                                1. For the cache generation graph, if the sequence length of input_ids is smaller than
                                max_seq_length, the KV cache will be zero padded to the max_seq_length so that it can
                                be used for token generation graph.
                                2. For the token generation graph, max_seq_length equals the KV cache's sequence length.
        """
        # Precision controls
        self.mixedp_attn = mixedp_attn
        self.fp32_logits = fp32_logits
        self.fp32_ln = fp32_ln
        self.fp32_skip_add = fp32_skip_add

        # Skip model initialization for compilation and speed up random initialization
        self.lazy_init = lazy_init

        # Used for zero padding on KV cache
        self.max_seq_length = max_seq_length

        # For SDPA
        self.use_segmented_softmax_attn = use_segmented_softmax_attn
        self.seg_softmax_block_size = seg_softmax_block_size

        # For O1HD
        self.use_plugin_heuristics = use_plugin_heuristics

    @classmethod
    def create(cls,
               *,
               sn_args: Optional[Dict[str, Any]] = None,
               original_config: PretrainedConfig,
               original_config_overrides: Optional[Dict[str, Any]] = None
               ) -> Union[PretrainedConfig, 'SNPretrainedConfig']:
        """
        Args:
            sn_args: These are the specific arguments for this cls SNPretrainedConfig subclass. These take precedence.
            original_config: This should be an instance of the specific model class's HF config.
            original_config_overrides: This will be applied to the original_config to override the specific values.
        Returns:
            An SNPretrainedConfig subclass converted from original_config.
        """
        uber_dict = {}
        for d in [original_config.to_dict(), original_config_overrides, sn_args]:
            if d is not None:
                uber_dict.update(d)

        return cls(**uber_dict)

    @classmethod
    def get_model_type_id(cls) -> str:
        """
        Returns: the type-string for the model to associate with the config class, e.g. 'snllama',
        """
        if cls.model_type == "" and (cls is not SNPretrainedConfig):
            raise NotImplementedError("Expected subclass of SNPretrainedConfig to have model_type defined in class.")

        return cls.model_type

    def update_from_config(self, config: PretrainedConfig):
        """
        Args:
            config: a subclass of PretrainedConfig object used to update the SambaNova PretrainedConfig subclass
        """
        self.update(config.to_dict())

    @abstractmethod
    def update(self, config_dict: Dict[str, Any]):
        # Due to:
        #    * multiple inheritance structure of SNLlamaConfig (and other such configs)
        #    * the update_from_config method expecting the update(..) method to exist,
        #       which it will if the SNXyZConfig class also inherits a PretrainedConfig, which it does.
        # We add the update method from the PretrainedConfig class here as an abstract method so that all
        # subclasses of SNPretrainedConfig must implement it. This is mostly for contract and documentation.
        pass

    @staticmethod
    def split_args(provided_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            provided_args: the args to split into SNPretrainedConfig args and non-SNPretrainedConfig args.

        Returns: (sn_args, non_sn_args) tuple of two Dict[str, Any]
        """
        sn_args = SNPretrainedConfig.get_sn_args(provided_args)
        non_sn_args = {k: v for k, v in provided_args.items() if k not in sn_args}

        return sn_args, non_sn_args

    @classmethod
    def get_sn_args(cls, provided_args: Dict[str, Any], exclude_args: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Get the arguments from provided_args that are also constructor arguments for the specific implementing subclass.
        'self' and 'kwargs' are always excluded, as well as all arguments specified in the exclude_args set.

        Note: !! __init__ must not contain *args else TypeError is raised.

        E.g: given a subclass of SNPretrainedConfig named Foo, where Foo.__init__(self, name, awesomeness, secret)
        calling:
          > Foo.get_valid_arguments({'a': 1, 'name':'fancy', 'secret': 'cookie'}, {'secret'})
              will return {'name':'fancy'}
          > Foo.get_valid_arguments({'a': 1, 'name':'fancy', 'awesomeness': 'bar'})
              will return {'name':'fancy', 'awesomeness': 'bar'}
        Args:
            provided_args: a dictionary with arguments we are interested in.
            exclude_args: a set of additional arguments to exclude (other than self and kwargs).

        Returns: the dictionary of args and values from provided_args where args
                 are non-excluded constructor arguments of cls.
        """
        if provided_args == {}:
            return {}

        all_args = cls.get_combined_sn_arg_names(exclude_args)
        valid_args = {arg: provided_args[arg] for arg in all_args if arg in provided_args}
        return valid_args

    @classmethod
    def get_combined_sn_arg_names(cls, exclude_args: Optional[Set[str]] = None) -> Set[str]:
        """
        Args:
            exclude_args: a set of additional arguments to exclude (other than self and kwargs).

        Returns:
            The combined argument names of the subclass and baseclass, except for exclude_args
        """
        def get_argument_map(specific_class: Type[SNPretrainedConfig]) -> Dict[str, str]:
            """
            For specific_class get the arguments and param string representation as a map representing
            the arguments to the __init__(self,...) constructor function.
                # e.g. {'self': 'self', 'c':, 'c=3', 'args': '*args', 'kwargs': '**kwargs'}
            Args:
                specific_class, the class to get the arguments list of
            Returns:
                The map of arg: <string representation> for the constructor of specific_class.
            """
            import inspect
            return {arg: str(param) for (arg, param) in inspect.signature(specific_class.__init__).parameters.items()}

        baseclass = SNPretrainedConfig
        is_baseclass = cls is baseclass
        subclass_args = get_argument_map(cls) if not is_baseclass else {}
        baseclass_args = get_argument_map(baseclass)

        # Prevent subclasses with *args for now, there does not seem to be a
        # meaningful use-case for *args in SNPretrainedConfig subclasses.
        if '*args' in subclass_args.values():
            raise TypeError(f'The constructor of {cls.__name__} must either:'
                            f' not be overloaded, or have a specific argument list + kwargs. Must not have *args.')

        # Matches the arguments in the constructor, including types and defaults that
        #  we do not want to expose as subclass SN model patch parameters.
        always_excluded = {
            'self',
            '**kwargs',
        }

        excluded_params: Set[str] = always_excluded.union(exclude_args if exclude_args is not None else set())

        filtered_baseclass_args = {key for (key, value) in baseclass_args.items() if value not in excluded_params}
        filtered_subclass_args = {key for (key, value) in subclass_args.items() if value not in excluded_params}
        all_arg_names = filtered_baseclass_args.union(filtered_subclass_args)
        return all_arg_names
