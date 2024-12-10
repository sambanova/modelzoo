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

import inspect
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from inspect import isclass
from typing import Any, Dict, Optional, Set, Tuple, Type, Union
from typing import get_args as typing_get_args

from transformers import PretrainedConfig

HUGGING_FACE_SPECIALS_TO_CONSTRUCTOR_ARGS_MAP = {
    # this is not a special arg, it is here to show how to add internal fields that should be ignored.
    '_dummy_ignored_': None,

    # this is not exported at all, @property _attn_implementation
    '_attn_implementation_internal': 'attn_implementation',

    # this is exported as _name_or_path (probably a bug)
    '_name_or_path': 'name_or_path',

    # yes, it maps to itself, but it is not exported
    '_commit_hash': '_commit_hash'
}
"""
This is a map from self.field_name to a constructor argument name, or None
A mapping to 'None' signifies that the field_name has no constructor argument, which means it will not
be included when extracting constructor arguments.
{
 'field_name': 'argument_name' or None
} 

There are special variables in the PretrainedConfig that are not exported as they are, or simply have internal names
that do not match the constructor argument. Also, some are not internally named the same as the @property accessors 
that differ from the constructor argument names and the member variable name.

For some reason HF are exporting the private field _name_or_path (probably a bug) 
even though there exists a @property 'name_or_path'.

'specials' is different from just the _x found in __dict__. We need to know what is not exported and how to 
use it when constructing a new PretrainedConfig.

I have chosen to include _name_or_path in specials, since the constructor arg is name_or_path.
"""

# Never reinstate these, use the original values from HF.
# This is used when exporting the hf subset config.
_SKIP_REINSTATE = {'model_type', 'architectures'}

# This is the key in the _sn_meta_data dict where we keep the dict of HF fields we want to keep a record of.
_META_DATA_HF_KEY = 'original_hf_fields'

# These are the HF fields we want to keep a record of
_ORIGINAL_HF_FIELDS = ['model_type', 'architectures']

# These are fields that we will ignore in certain tests that automate creation.
_TEST_IGNORE_FIELDS = {'source_hf_config', '_sn_meta_data'}


class SNIllegalSuperClassInitError(TypeError):
    pass


class SNPretrainedConfig(ABC):
    model_type: str = ""

    def __init__(
            self,
            mixedp_attn: bool = False,
            fp32_logits: bool = False,
            fp32_ln: bool = False,
            fp32_skip_add: bool = False,
            lazy_init: bool = False,
            run_early_tp: bool = False,
            use_plugin_heuristics: bool = False,
            has_classifier_layer: bool = True,
            use_segmented_softmax_attn: bool = False,
            seg_softmax_block_size: Optional[int] = None,
            max_seq_length: Optional[int] = None,
            max_tokens_to_generate: Optional[int] = None,
            # ------------------
            # Special meta data that we want to save (and load)
            _sn_meta_data: Optional[Dict[str, Any]] = None,
            # Needed by init_superclasses(..), must be defined in this constructor,
            #  if not supplied default superclass instance is used.
            source_hf_config: Optional[PretrainedConfig] = None,  #noqa
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
                run_early_tp:  If true, distribute computation across multiple sockets early in compilation.
                use_plugin_heuristics: If true, use o1 HD heuristics for customized mapping.
                has_classifier_layer: If False, denotes the model is used for embedding generation and doesn't have classifier layer. Used to customize mapping decisions.
                use_segmented_softmax_attn: If true, use SDPA on RDU. To be deprecated 
                                            after torch 2.0 upgrade
                seg_softmax_block_size: If set, force a key-value block size in segmented softmax attention
                                        in SDPA.
                max_seq_length: The sequence length of the KV cache used by the token generation graph. It is the maximum
                                number of tokens one can generate including the input prompts.
                                1. For the cache generation graph, if the sequence length of input_ids is smaller than
                                max_seq_length, the KV cache will be zero padded to the max_seq_length so that it can
                                be used for the token generation graph.
                                2. For the token generation graph, max_seq_length equals the KV cache's sequence length.
                max_tokens_to_generate: The maximum number of tokens to generate. In the case of spec decoding, the token gen graph will
                                        only generate K tokens therefore to reduce the cost of transferring the result to host the 
                                        generated_tokens tensor will have a max size of BSxK.
                _sn_meta_data: Information needed to recreate the original source config
                source_hf_config: the Hugging Face config that we used when converting to SambaNova config. Optional
        """
        if source_hf_config is None:
            hf_super_class = get_hf_super_class(self.__class__)
            # Super class may be None in some tests, when you create subclasses that are not dual inherited
            source_hf_config = hf_super_class() if hf_super_class is not None else None

        provided_meta_data = _sn_meta_data if _sn_meta_data is not None else {}
        self._sn_meta_data = SNPretrainedConfig._init_meta_data(provided_meta_data, source_hf_config)

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
        self.run_early_tp = run_early_tp
        self.has_classifier_layer = has_classifier_layer

        # For Spec Decoding
        if max_tokens_to_generate is not None and max_seq_length is not None:
            assert (max_tokens_to_generate <= max_seq_length and max_tokens_to_generate > 0), (
                "max_tokens_to_generate must be less than or equal to max_seq_length, "
                "and it should be greater than 0, but got {} and {}, respectively.".format(
                    max_tokens_to_generate, max_seq_length))
        self.max_tokens_to_generate = max_tokens_to_generate

    @staticmethod
    def init_superclasses(*,
                          subclass_self: Union[PretrainedConfig, 'SNPretrainedConfig'],
                          kwargs_dict: Dict[str, Any],
                          sn_overrides: Optional[Dict[str, Any]] = None,
                          hf_overrides: Optional[Dict[str, Any]] = None):
        """
        Calls the __init__ functions of the subclass_self's super-classes and handles
        SNPretrained special handling for HF overrides.

        Args:
            subclass_self: the reference to the subclass object doing the init
            kwargs_dict: the kwargs to apply as SN args and HF overrides.
            sn_overrides: The overrides the subclass has hard-coded for the SambaNova SNPretrainedConfig super class
            hf_overrides: The overrides the subclass has hard-coded for the Hugging Face super class
        """
        hf_super_class = get_hf_super_class_or_raise(subclass_self.__class__)

        sn_overrides = sn_overrides if sn_overrides is not None else {}
        hf_overrides = hf_overrides if hf_overrides is not None else {}

        (sn_base_args, hf_base_args) = SNPretrainedConfig.split_args(kwargs_dict)

        # --------
        # Apply overrides to the source hf config
        source_hf_config: Optional[PretrainedConfig] = sn_base_args.get('source_hf_config', None)
        source_args = get_non_default_external_hf_args(source_hf_config, is_recursive=True)
        special_args = get_non_default_hf_special_args(source_hf_config, is_recursive=True)
        hf_args_with_override = {**source_args, **special_args, **hf_base_args, **hf_overrides}

        # Override attn_implementation if not set by users
        if 'attn_implementation' not in hf_args_with_override:
            hf_args_with_override['attn_implementation'] = 'eager'

        if source_hf_config is not None:
            # Avoid duplicate in input to init below
            sn_base_args.pop('source_hf_config', None)
            updated_source_hf_config = apply_hf_config_overrides(config=source_hf_config,
                                                                 overrides=hf_args_with_override)
        else:
            # Create default instance with overrides, do not include overridden
            #  model_type etc. defined in _ORIGINAL_HF_FIELDS
            base_default: Dict[str, Any] = vars(hf_super_class())
            hf_args_for_default = {
                k: v
                for k, v in hf_args_with_override.items()
                if not (k in _ORIGINAL_HF_FIELDS or base_default.get(k, None) == v)
            }
            updated_source_hf_config = hf_super_class(**hf_args_for_default)

        # --------
        # Initialize the (expected, singular) Hugging Face Pretrained Config
        hf_super_class.__init__(subclass_self, **hf_args_with_override)

        # --------
        # Initialize the SNPretrainedConfig superclass
        sn_args_with_overrides = {**sn_base_args, **sn_overrides}
        SNPretrainedConfig.__init__(subclass_self, **sn_args_with_overrides, source_hf_config=updated_source_hf_config)

    @staticmethod
    def _init_meta_data(provided_meta_data: Dict[str, Any], source_hf_config: Optional[PretrainedConfig]):
        if provided_meta_data != {}:
            # provided meta data overrides exiting information
            sn_meta_data = deepcopy(provided_meta_data)
            return sn_meta_data

        # source_hf_config may be an SN config if we self-transform in ConfigurationTransformer, then we
        # don't want to take the meta_data fields from source_hf_config, since they will be sn variants.
        # Instead, copy the _meta_data info.
        if isinstance(source_hf_config, SNPretrainedConfig):
            sn_meta_data = deepcopy(source_hf_config._sn_meta_data)
            return sn_meta_data

        sn_meta_data = {}
        if source_hf_config is not None:
            sn_meta_data[_META_DATA_HF_KEY] = {}
            for field in _ORIGINAL_HF_FIELDS:
                if hasattr(source_hf_config, field):
                    sn_meta_data[_META_DATA_HF_KEY][field] = deepcopy(getattr(source_hf_config, field))
        return sn_meta_data

    @staticmethod
    def init_sub_config(sub_config: Union['SNPretrainedConfig', Dict[str, Any], None],
                        sub_config_class: Type['SNPretrainedConfig']) -> 'SNPretrainedConfig':
        """
        Possibly convert a sub-config dict to a proper SNPretrainedConfig object.
        This is needed due to how config file parsing is done. In those cases sub-configs will be pure dicts.
        
        Args:
            sub_config: the sub config input from the constructor of the SNPretrainedConfig subclass.
                        This must either be of the specific SNPretrainedConfig subclass matching 'sub_config_class'
                        type or a Dict[str, Any]. If it is None, a default object will be created.
            sub_config_class: the class type of the sub-config to construct in the event that sub_config is a Dict
                              This type is also used to verify the type of sub_config if it is not a Dict.
        Returns:
            The sub_config as a proper SNPretrainedConfig object instance of type 'sub_config_class'.
            If sub_config is None, a default instance of 'sub_config_class' is created.

        Raises:
            TypeError if sub_config is not of the expected type
            ValueError if the sub_config as dict has a 'model_type' and it is not the same as
                       sub_config_class.model_type.
        """
        if sub_config is None:
            return sub_config_class()

        def is_correct_dict_type(possible_dict):
            if not isinstance(possible_dict, Dict):
                return False
            return all(isinstance(k, str) for k in possible_dict)

        if type(sub_config) != sub_config_class and not is_correct_dict_type(sub_config):
            raise TypeError(f'Sub-config must be a SNPretrainedConfig subclass or a Dict[str,Any], '
                            f'found type={type(sub_config)}')

        if isinstance(sub_config, Dict):
            model_type = sub_config.get('model_type', None)
            if model_type is not None and model_type != sub_config_class.model_type:
                raise ValueError(f'The sub_config dict model_type does not match model_type of '
                                 f'{sub_config_class.__name__}. Expected "{sub_config_class.model_type}", '
                                 f'found "{model_type}"')
            return sub_config_class(**sub_config)

        return sub_config

    def to_dict(self) -> Dict[str, Any]:
        """
        Do not call this, it will raise Error. This is a guard against incorrect inheritance order.
        """
        raise NotImplementedError('This method must never be called, then the inheritance order is wrong.')

    @classmethod
    def create(
            cls,
            *,
            sn_args: Optional[Dict[str, Any]] = None,
            original_config: PretrainedConfig,
    ) -> Union[PretrainedConfig, 'SNPretrainedConfig']:
        """
        Args:
            sn_args: Specific arguments for this SNPretrainedConfig subclass. These take precedence.
            original_config: An instance of the specific model class's Hugging Face config.
        Returns:
            An SNPretrainedConfig subclass converted from the original_config.
        """
        sn_args = sn_args if sn_args is not None else {}
        original_config_copy = deepcopy(original_config)

        return cls(source_hf_config=original_config_copy, **sn_args)

    @classmethod
    def get_model_type_id(cls) -> str:
        """
        Returns: the type string to associate with the config class for the model, e.g. 'snllama',
        """
        if cls.model_type == "" and (cls is not SNPretrainedConfig):
            raise NotImplementedError(f'A subclass of SNPretrainedConfig [{cls}] must have the '
                                      f'\'model_type\' class variable defined.')

        return cls.model_type

    def update_from_config(self, config: PretrainedConfig):
        """
        Args:
            config: a subclass of the PretrainedConfig object used to update the SambaNova PretrainedConfig subclass
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

        Returns: (sn_args, non_sn_args) tuple of two Dict[str, Any] with model_type removed
        """
        sn_args = SNPretrainedConfig.get_sn_args(provided_args)
        # Note that we must remove model_type, since providing it from provided_args will shadow the
        # SNPretrained subclass' class-variable with an instance variable that will be the HF model_type
        non_sn_args = {k: v for k, v in provided_args.items() if k not in sn_args}

        return sn_args, non_sn_args

    @classmethod
    def get_sn_args(cls, provided_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all the arguments from 'provided_args' that are also constructor arguments for the specific implementing
        subclass or the 'SNPretrainedConfig' baseclass. 'self', 'kwargs', are always excluded.

        Note: !! __init__ must not contain *args else TypeError is raised.

        E.g: given a subclass of SNPretrainedConfig named Foo, where Foo.__init__(self, name, awesomeness, secret)
        calling:
          > Foo.get_valid_arguments({'a': 1, 'name':'fancy', 'secret': 'cookie'})
              will return {'name':'fancy', 'secret': 'cookie'}
        Args:
            provided_args: a dictionary with arguments we are interested in.

        Returns: the dictionary of args and values from provided_args where args
                 are constructor arguments of cls, excluding self and kwargs
        """
        if provided_args == {}:
            return {}

        all_args = cls._get_combined_sn_arg_names()
        valid_args = {arg: provided_args[arg] for arg in all_args if arg in provided_args}
        return valid_args

    @classmethod
    def _get_combined_sn_arg_names(cls) -> Set[str]:
        """
        Returns:
            The combined argument names of the subclass and baseclass
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
            return {arg: str(param) for (arg, param) in inspect.signature(specific_class.__init__).parameters.items()}

        baseclass = SNPretrainedConfig
        is_baseclass = cls is baseclass
        subclass_args = get_argument_map(cls) if not is_baseclass else {}
        baseclass_args = get_argument_map(baseclass)

        # Prevent subclasses with *args for now, there does not seem to be a
        # meaningful use case for *args in SNPretrainedConfig subclasses.
        if '*args' in subclass_args.values():
            raise TypeError(f'The constructor of {cls.__name__} must either:'
                            f' not be overloaded, or have a specific argument list + kwargs. Must not have *args.')

        # Matches the arguments in the constructor, including types and defaults that
        # we do not want to expose as subclass SN model patch parameters.
        always_excluded = {
            'self',
            '**kwargs',
        }

        filtered_baseclass_args = {key for (key, value) in baseclass_args.items() if value not in always_excluded}
        filtered_subclass_args = {key for (key, value) in subclass_args.items() if value not in always_excluded}
        all_arg_names = filtered_baseclass_args.union(filtered_subclass_args)
        return all_arg_names

    @classmethod
    def get_sn_subclass_nested_config_args(cls) -> Dict[str, ABCMeta]:
        """
        Returns:
            The explicit arguments of the subclass's constructor only (not the baseclass)
            that are of SNPretrainedConfig types, that is, the sub-config fields in the constructor.
        """
        def is_sn_config_subclass(clazz):
            if isclass(clazz) and issubclass(clazz, SNPretrainedConfig):
                return True

            types = typing_get_args(clazz)
            return any(isclass(t) and issubclass(t, SNPretrainedConfig) for t in types)

        sn_args = cls._get_sn_subclass_args()
        sn_sub_configs = {name: clazz for name, clazz in sn_args.items() if is_sn_config_subclass(clazz)}
        return sn_sub_configs

    @classmethod
    def _get_sn_subclass_args(cls) -> Dict[str, ABCMeta]:
        """
        Returns:
            The explicit arguments of the subclass only (not the baseclass). Excludes (self, *args and **kwargs)
            This is a dict mapping argument name to the argument type.
        """
        if cls is SNPretrainedConfig:
            raise ValueError('Method called on base class is undefined. Must be called on sub-class')

        excluded_parameters = {
            'self',
            '*args',
            '**kwargs',
        }
        args = {name: parameter for name, parameter in inspect.signature(cls.__init__).parameters.items()}
        filtered_args = {
            name: parameter.annotation
            # use of str(parameter) is intentional. Converts the Parameter object to '**kwargs' and '*args'
            for (name, parameter) in args.items() if str(parameter) not in excluded_parameters
        }
        return filtered_args

    def get_hf_config_subset(self) -> PretrainedConfig:
        """
        Returns:
            The Hugging Face config subset of parameters as they are after construction of the SN config subclass.
            Retains the original HF parameters for model_type and architecture, this applies recursively for
            nested configs.
        """
        # Recurse and update the source HF config, override all but _SKIP_REINSTATE fields.
        recreated_hf_config = SNPretrainedConfig._recursive_update_hf_config_with_overrides(self, _SKIP_REINSTATE)
        return recreated_hf_config

    @staticmethod
    def _recursive_update_hf_config_with_overrides(sn_config: 'SNPretrainedConfig',
                                                   skip_reinstate_fields: Set[str]) -> PretrainedConfig:
        """
            Recursively apply changes from sn_config to the hf_config, except some "special" changes
            to fields defined in _SKIP_REINSTATE. Changes are applied inplace and will modify the internal
            hugging face config instance.

            Returns:
                the Pretrained config
        """

        # Create a new default instance of the HF superclass
        hf_config = get_hf_super_class_or_raise(sn_config.__class__)()

        sn_sub_configs = sn_config.get_sn_subclass_nested_config_args()

        # Recurse if sub-configs
        for field_name, config_type in sn_sub_configs.items():
            sub_config = getattr(sn_config, field_name)
            sub_hf_config = SNPretrainedConfig._recursive_update_hf_config_with_overrides(
                sub_config, skip_reinstate_fields)
            # Replace the hf_config's sub_config with the updated one
            setattr(hf_config, field_name, sub_hf_config)

        # Base case
        #  Find fields that need to be updated for hf config based on the current sn config.
        #  Do not update sub-config fields at this level, they were already updated recursively
        skip_fields_and_sub_config = set(sn_sub_configs.keys()).union(skip_reinstate_fields)
        SNPretrainedConfig._update_hf_overrides_inplace(sn_config, hf_config, skip_fields_and_sub_config)
        return hf_config

    @staticmethod
    def _update_hf_overrides_inplace(sn_config: 'SNPretrainedConfig', hf_config: PretrainedConfig,
                                     skip_reinstate_fields: Set[str]):
        sn_keys = {key for key in vars(sn_config).keys() if key not in skip_reinstate_fields}
        for key in sn_keys:
            if hasattr(hf_config, key):
                current_value = getattr(hf_config, key)
                sn_value = getattr(sn_config, key)
                if current_value != sn_value:
                    setattr(hf_config, key, deepcopy(sn_value))

        meta_data = sn_config._sn_meta_data if sn_config._sn_meta_data is not None else {}
        for key, value in meta_data.get(_META_DATA_HF_KEY).items():
            setattr(hf_config, key, deepcopy(value))


# -------------------------------- End of SNPretrainedConfig --------------------------------


def apply_hf_config_overrides(*, config: PretrainedConfig, overrides: Dict[str, Any]) -> PretrainedConfig:
    """
    Recursively apply 'overrides' to 'config'. In the nested case, overrides must match the nested structure in
    order to be applied recursively. 'overrides' is applied as is with not filtering whatsoever.

    Args:
        config: the Hugging Face PretrainedConfig to override, may contain nested configs.
        overrides: The overrides, may contain a nested structure of overrides. May be {} but not None.

    Returns:
        A new config object of the same type as 'config' with overrides applied recursively.
    """
    sub_configs = {name: clazz for name, clazz in vars(config).items() if isinstance(clazz, PretrainedConfig)}

    collected_sub_configs = {}

    # Recurse if sub-configs
    for field_name, config_type in sub_configs.items():
        sub_config_to_override = getattr(config, field_name)
        sub_overrides = overrides.get(field_name, {})
        overrides.pop(field_name, None)
        overridden_config = apply_hf_config_overrides(config=sub_config_to_override, overrides=sub_overrides)
        collected_sub_configs[field_name] = overridden_config

    # Base case: apply overrides to copy of config
    #
    # WARNING: You may think this is a strange piece of code, and you are correct.
    #   We must construct a new config using the to_dict as well as overrides, since just calling _.update() will not
    #   include all parameters and may incorrectly add parameters to the dict that we do not want.
    #   Take for example attn_implementation: since the member is '_attn_implementation_internal'
    #   accessible by the '_attn_implementation' property method, but the constructor takes 'attn_implementation'.
    #   There are probably others as well.
    #   If the override does not contain attn_implementation, but the original config does, then we loose that
    #   when we construct the object by constructor, since there is no way to export the property from 'config'.
    #   We need special magic to compose the new arguments to the constructor.
    #   Running __init__(..) will overwrite the _attn_implementation_internal and others... because kwargs.pop(xxx, None)
    config_update = deepcopy(config)
    non_default_args = get_non_default_external_hf_args(config_update)
    non_default_special_args = get_non_default_hf_special_args(config)
    constructor_args = {**non_default_args, **non_default_special_args, **overrides, **collected_sub_configs}
    config_update.__init__(**constructor_args)  # overrides are always constructor arguments, run init again

    return config_update


def remove_sn_args_from_original_config_overrides(clazz: Type[SNPretrainedConfig],
                                                  overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Removed entries in overrides that are defined in SNPretrainedConfig or a subclass
    Args:
        clazz: a SNPretrainedConfig subclass
        overrides: the overrides we want to filter, this is normally original HF config overrides

    Returns:
        A new dict where relevant entries are removed, or {} if overrides is None
    """
    if overrides is None or overrides == {}:
        return {}

    overrides_copy = deepcopy(overrides) or {}
    to_be_removed_args = clazz.get_sn_args(overrides)
    for p in to_be_removed_args.keys():
        overrides_copy.pop(p, None)

    return overrides_copy


def get_hf_super_class(sn_subclass: Type[SNPretrainedConfig]) -> Optional[Type[PretrainedConfig]]:
    """
    Args:
        sn_subclass: the SNPretrained subclass to get the Hugging Face config superclass from.

    Returns:
        The Hugging Face super class of 'subclass_self'
    Raises:
        SNIllegalInheritanceError if not exactly 1 "pure" Hugging Face superclass is found.
                                  By pure we mean it is not also a subclass of SNPretrainedConfig
    """
    super_classes = [
        base for base in sn_subclass.__bases__
        # hf super-class must not be SNPretrainedConfig. -> infinite recursion
        if issubclass(base, PretrainedConfig) and not issubclass(base, SNPretrainedConfig)
    ]

    if len(super_classes) == 1:
        return super_classes[0]

    return None


def get_hf_super_class_or_raise(sn_subclass: Type[SNPretrainedConfig]) -> Type[PretrainedConfig]:
    """
    Args:
        sn_subclass: the SNPretrained subclass to get the Hugging Face config superclass from.

    Returns:
        The Hugging Face super class of 'subclass_self'
    Raises:
        SNIllegalInheritanceError if not exactly 1 "pure" Hugging Face superclass is found.
                                  By pure we mean it is not also a subclass of SNPretrainedConfig
    """
    super_class = get_hf_super_class(sn_subclass)
    if super_class is None:
        raise SNIllegalSuperClassInitError(f'Expected this class {sn_subclass.__class__.__name__} '
                                           f'to be subclass of exactly 1 PretrainedConfig (or subclass thereof).')
    return super_class


def get_non_default_hf_special_args(source_hf_config: PretrainedConfig, is_recursive=False) -> Dict[str, Any]:
    """
    Get the "special" internals that do not match the name of the constructor
    arguments or are not exported when calling to_dict().

    Args:
        source_hf_config: a Hugging Face config from which to extract "special" overridden values
                          as constructor arguments.
        is_recursive: handle recursive sub-config extraction, default False

    Returns:
        A dict with {argument_name: value,...} that can be supplied as constructor arguments to source_hf_config's class
    """
    if not isinstance(source_hf_config, PretrainedConfig):
        return {}

    hf_class = source_hf_config.__class__
    defaults = vars(hf_class())
    internals = vars(source_hf_config)

    internals_no_defaults = {}
    for k, v in internals.items():
        if isinstance(v, PretrainedConfig):  # possibly recurse for sub_config
            sub_specials = get_non_default_hf_special_args(v) if is_recursive else {}
            if sub_specials != {}:  # either no recursion or sub_specials is actually empty
                internals_no_defaults[k] = sub_specials
        elif HUGGING_FACE_SPECIALS_TO_CONSTRUCTOR_ARGS_MAP.get(k, None) is not None:
            if k not in defaults or v != defaults.get(k):
                internals_no_defaults[k] = v

    as_args = {HUGGING_FACE_SPECIALS_TO_CONSTRUCTOR_ARGS_MAP.get(k, k): v for k, v in internals_no_defaults.items()}
    return as_args


def get_non_default_external_hf_args(source_hf_config: Optional[PretrainedConfig],
                                     is_recursive: bool = False) -> Dict[str, Any]:
    """
    Args:
        source_hf_config: the PretrainedConfig to get the non-default values from
        is_recursive: do it recursively if and only if this is true.

    Returns:
        The Dict with key,values in the to_dict() representation (can be sent to the constructor) that are NOT defined
        as "special" args according to HUGGING_FACE_SPECIALS_TO_CONSTRUCTOR_ARGS_MAP.
    """
    if not isinstance(source_hf_config, PretrainedConfig):
        return {}

    hf_class = source_hf_config.__class__
    defaults = hf_class().to_dict()
    externals = source_hf_config.to_dict()

    externals_no_defaults = {}
    for k, v in externals.items():
        possible_sub_config = getattr(source_hf_config, k)
        if isinstance(possible_sub_config, PretrainedConfig):  # possibly recurse for sub_config
            sub_specials = get_non_default_external_hf_args(possible_sub_config, True) if is_recursive else {}
            if sub_specials != {}:  # either no recursion or sub_specials is actually empty
                externals_no_defaults[k] = sub_specials
        elif k not in HUGGING_FACE_SPECIALS_TO_CONSTRUCTOR_ARGS_MAP:
            if k not in defaults or v != defaults.get(k):
                externals_no_defaults[k] = v

    return externals_no_defaults
