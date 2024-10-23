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

import functools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Type

import torch
import torch.nn.functional as F
from sambanova_modelzoo.libs.nlp.core.generation.sampling import SamplingMethod

from .attention_mask_utils import get_collapsed_article_attention_mask
from .token_utils import ARTICLE_SEP_TOKEN_TYPE_ID, PADDING_TOKEN_TYPE_ID

""" Used by both JIT (compile and run) and Samba runtime so this file cannot import samba """
# TODO: Name of this file is clm_runtime. What does that mean?


class PytorchGenerationInputNames(Enum):
    """
    Hugging Face's model.generate() function redirects its inputs to model() forward function. To work with
    model.generate(), the input names defined here also need to be redirected to the model() function as keyword arguments.
    For example, last_token_index is a Model Zoo specific argument that also exists in the SNLlamaForCausalLM class's
    forward() function call signature.

    All the string names here need to match the common SambaNova model's keyword arguments for the forward() function.
    """
    input_ids = 'input_ids'
    attention_mask = 'attention_mask'
    last_token_index = 'last_token_index'
    logits = 'logits'
    last_hidden_states = 'last_hidden_states'
    generated_tokens = 'generated_tokens'
    generated_index = 'generated_index'
    temperature = 'temperature'
    top_k = 'top_k'
    top_p = 'top_p'
    pre_generated_randoms = 'pre_generated_randoms'
    repetition_penalty = 'repetition_penalty'
    token_count = 'token_count'


class SambaGenerationInputNames:
    """
    Customized names used to match between compilation and runtime. During compilation, the input symbols
    names are burned into the PEF. During runtime, the exact name needs to be used to set to the correct symbol address
    in the PEF.

    Input names have to be unique during tracing between cache_gen and token_gen call schedule inside the same PEF.
    However, during runtime, it's the region name that's matches the symbol name in the PEF. Different tensors with
    different names can share the same region name, thus the same symbol address in the PEF. The name listed are all
    region names that can be used to retrieve tensor from RDU at runtime.
    By default, if region name is not specified, it is the same as tensor name.
    # TODO: "have to be unique during tracing" or "have to be unique during compilation"

    input_ids: Different region names between the cache_gen and token_gen graph because they have different shapes
    attention_mask: Different region names between cache_gen and token_gen graph because they have different shapes
    last_token_index: Same region name shared between cache_gen and token_gen graph since they have the same shape
    """
    def deduplicate(self, name: str, **kwargs) -> str:
        """
        Args:
            name: Original name.
            kwargs: Keyword argments that are used to deduplicate the name.
        Returns:
            A unique name across different graphs in the same PEF.
        """
        for k, v in kwargs.items():
            name += f'_{k}_{v}'
        return name

    def cache_gen_input_ids(self, static_seq_length: int) -> str:
        """ Returns input_ids name for cache_gen graph, unique across graphs """
        return self.deduplicate(PytorchGenerationInputNames.input_ids.value,
                                consume_cache=False,
                                static_seq_length=static_seq_length)

    def token_gen_input_ids(self, max_seq_length: int) -> str:
        """ Returns input_ids name for token_gen_graph, unique across graphs """
        return self.deduplicate(PytorchGenerationInputNames.input_ids.value,
                                consume_cache=True,
                                max_seq_length=max_seq_length)

    @property
    def attention_mask(self) -> str:
        """ Returns attention_mask name, unique across cache_gen and token_gen graphs """
        return PytorchGenerationInputNames.attention_mask.value

    @property
    def last_token_index(self) -> str:
        """ Returns last_token_index name, shared between cache_gen and token_gen graphs """
        return PytorchGenerationInputNames.last_token_index.value

    @property
    def logits(self) -> str:
        """ Retrurns the logits input name for the postprocessing graph """
        return PytorchGenerationInputNames.logits.value

    @property
    def last_hidden_states(self) -> str:
        """ Retrurns the last_hidden_states input name for fast the postprocessing graph """
        return PytorchGenerationInputNames.last_hidden_states.value

    @property
    def generated_tokens(self) -> str:
        """
        Returns the generated_tokens input and output name for the postprocessing graph. It will be updated in-place to hold
        all generated tokens.
        """
        return PytorchGenerationInputNames.generated_tokens.value

    @property
    def generated_index(self) -> str:
        """ Returns the generated_index tensor name. generated_index is used for generated_tokens update """
        return PytorchGenerationInputNames.generated_index.value

    @property
    def temperature(self) -> str:
        return PytorchGenerationInputNames.temperature.value

    @property
    def top_k(self) -> str:
        return PytorchGenerationInputNames.top_k.value

    @property
    def top_p(self) -> str:
        return PytorchGenerationInputNames.top_p.value

    @property
    def pre_generated_randoms(self) -> str:
        return PytorchGenerationInputNames.pre_generated_randoms.value

    @property
    def repetition_penalty(self) -> str:
        return PytorchGenerationInputNames.repetition_penalty.value

    @property
    def token_count(self) -> str:
        return PytorchGenerationInputNames.token_count.value


@dataclass(frozen=True)
class SambaPretrainInputNames:
    """
    Customized input names used to match between compilation and runtime. During compilation, the input symbols
    names are burned into the PEF. During runtime, the exact name needs to be used to set to the correct symbol address
    in the PEF.
    """
    input_ids: str = 'input_ids'
    attention_mask: str = ('attention_mask_collapsed_1', 'attention_mask_collapsed_2')
    labels: str = 'labels'

    def __getitem__(self, item):
        return getattr(self, item)


class SambaGraphNames:
    @staticmethod
    @functools.lru_cache
    def cache_gen(static_sequence_length: int) -> str:
        """
        Args:
            static_sequence_length: The sequence length for the input ids to pad to. This can be shorter than max_seq_length.
        Returns:
            Name of graph generated by the cache generation pass. This name is compiled into the PEF and then selected
            at runtime to run this particular graph schedule.
        """
        return 'model_nocache_' + str(static_sequence_length) + '_fwd'

    @staticmethod
    @functools.lru_cache
    def token_gen(max_seq_length: int) -> str:
        """
        Args:
            max_seq_length: The sequence length of static KV cache. The maximum number of tokens that can be generated
            is (max_seq_length - prompt_input_length).
        Returns:
            Name of the graph generated by the continuous token generation pass. This name is compiled into the PEF and then
            selected at runtime to run this particular graph schedule.
        """
        return 'model_cache_' + str(max_seq_length) + '_fwd'

    @staticmethod
    @functools.lru_cache
    def postprocess(sampling_method: SamplingMethod) -> str:
        """
        Postprocessing includes:
        1. sampling (including greedy) on the logits for tokens
        2. Update token_gen attention mask vector to set 1 at last_token_index positions
        3. Update last_token_index to increment by 1
        We fuse all above in the same graph for compiler optimization.
        Returns:
            Name of postprocessing graph.
        """
        return 'postprocess_' + sampling_method.value

    @staticmethod
    @functools.lru_cache
    def lm_head() -> str:
        """
        Returns:
            Name of lm_head graph.
        """
        return 'lm_head'


def expensive_error_if(error_type: Type[Exception], msg: str, condition: Callable[..., bool], *args, **kwargs):
    # TODO: have a better global control
    if os.getenv('SKIP_SANITY_CHECK', 0):
        return
    if condition(*args, **kwargs):
        raise error_type(msg)


def is_left_padded(input_ids: torch.Tensor, pad_token_id: int) -> bool:
    """
    Args:
        input_ids: Input token ids
        pad_token_id: Padding token id
    Returns:
        True if any row in `input_ids` is left padded. A full row of padding tokens does not count as left padded since it
        can be used to pad batch_dim for a static batch size PEF.
    """
    starts_with_pad_row = input_ids[:, 0] == pad_token_id
    if not torch.any(starts_with_pad_row):
        return False
    return torch.any(input_ids[starts_with_pad_row, :] != pad_token_id)


def counted(f: Callable[..., Any]):
    """
    Records the number of times that f is called.
    """
    def wrapped(*args, **kwargs):
        res = f(*args, **kwargs)
        wrapped.calls += 1
        return res

    wrapped.calls = 0
    return wrapped


class CachedInferenceRuntime:
    model_to_runtime: Dict[Type[torch.nn.Module], Type['CachedInferenceRuntime']] = {}

    def __init__(self,
                 max_seq_length: int,
                 pad_token_id: int,
                 sliding_window: Optional[int] = None,
                 runner: str = 'pytorch'):
        """
        Base class for preprocessing inputs to run cached inference. The class is stateful in the sense that first
        preprocess_inputs is called for cache generation and then many token generation calls follow.
        preprocess_inputs can only be called (max_seq_length - min_prompt_len) times.

        This state machine makes the following assumptions:
        1. The prompt is right padded (left aligned).
        2. The token generation phase only sends seqlen=1, that is, the last generated tokens, as input id.
        3. The model has to take into account that the input id is right-padded.

        Args:
            max_seq_length: The maximum sequence length of the token generation graph.
            pad_token_id: Padding token id.
            sliding_window: Window size for sliding window attention. It's used to focus only on the past `sliding_window`
                            number of tokens.
            runner: The runner of the processed inputs, choices are ['pytorch' (default), 'samba'].
                    If runner is `pytorch`, the returned string key of the tensor has to match the model forward call signature.
                    If runner is `samba`, the returned string key does not have to match the model forward call signature.
                    Instead, the string key of the tensor has to match the compile time input symbol names.
        """
        if pad_token_id is None:
            raise ValueError("Please set a valid pad_token_id")
        self.max_seq_length: int = max_seq_length
        self.pad_token_id: int = pad_token_id
        self.sliding_window: int = sliding_window
        self._runner: str = None

        self.runner = runner

        # The initial input ids, with right padding (left alignment), only set once
        self._prompt_tokens: torch.Tensor = None
        # ====================inputs=====================
        # The indices of the last non-pad token. It's set once from the attention_mask to the right-padded prompt. Later
        # on, it is incremented by one each time.
        self.last_token_index: torch.Tensor = None
        self.input_ids: torch.Tensor = None
        self.attention_mask: torch.Tensor = None

        # Input names
        self.input_names = SambaGenerationInputNames()

        # Graph names used by Samba
        self.graph_names = SambaGraphNames()

    def __init_subclass__(cls, model: Type[torch.nn.Module] = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if model is None:
            raise RuntimeError('model needs to be specified to register CachedInferenceRuntime')
        CachedInferenceRuntime.model_to_runtime[model] = cls

    @classmethod
    def get_registered_runtime(cls, model_type: Type[torch.nn.Module]) -> 'CachedInfereceRuntime':
        """
        Args:
            model_type: The class name of the SNModel to be queried for.
        Returns:
            The registered class of the correponding ModelRuntime.
        """
        if model_type not in cls.model_to_runtime:
            raise ValueError(
                f"{model_type} is not registered with CachedInferenceRuntime. Registered are {cls.model_to_runtime.keys()}"
            )
        return cls.model_to_runtime[model_type]

    @property
    def runner(self) -> str:
        """
        Returns the runner mode. It's either `samba` or `pytorch`.
        """
        return self._runner

    @runner.setter
    def runner(self, val: str):
        """
        Sets the runner mode.
        """
        valid_runners = ['pytorch', 'samba']
        if val not in valid_runners:
            raise ValueError(f"Invalid runner {val}, must be one of {valid_runners}")
        self._runner = val

    @property
    def prompt_tokens(self) -> torch.Tensor:
        """
        Returns the input prompts.
        """
        return self._prompt_tokens

    @prompt_tokens.setter
    def prompt_tokens(self, prompts: torch.Tensor):
        """
        Sets the input prompts. You can only do it once, otherwise a RuntimeError will be raised.
        """
        if self.prompt_tokens is None:
            if prompts.shape[1] >= self.max_seq_length:
                raise ValueError("Prompts should be shorter than the compiled seq length")
            self._prompt_tokens = prompts
        else:
            raise RuntimeError(f"prompt_tokens has can only be set once, existed: {self.prompt_tokens}: new: {prompts}")

    def graph_to_call(self, cache_gen: bool, static_seq_length: int) -> str:
        """
        Args:
            cache_gen: Whether to get cache generation graph name.
            static_seq_length: The sequence length of the input_ids to be padded to for the cache_gen graph.
        Returns:
            The graph schedule name to call to process the inputs. Samba compiles the cache_gen and token_gen graphs into
            the same PEF as different schedules. This graph name is determined during compilation and written inside the
            PEF as the call schedule ID. At runtime, the name call be used to run a specific part of the PEF (call schedule),
            as either cache_gen or token_gen.
        """
        if cache_gen:
            return self.graph_names.cache_gen(static_seq_length)
        else:
            return self.graph_names.token_gen(self.max_seq_length)

    def input_ids_name(self, token_gen: Optional[bool] = None, static_seq_length: Optional[int] = None) -> str:
        """
        Returns tensor name of iput_ids.
        """
        if self.runner == 'pytorch':
            return PytorchGenerationInputNames.input_ids.value
        else:
            if token_gen:
                return self.input_names.token_gen_input_ids(self.max_seq_length)
            else:
                return self.input_names.cache_gen_input_ids(static_seq_length)

    @property
    def attention_mask_name(self) -> str:
        """
        Returns tensor name of attention_mask.
        """
        if self.runner == 'pytorch':
            return PytorchGenerationInputNames.attention_mask.value
        else:
            return self.input_names.attention_mask

    @property
    def last_token_index_name(self) -> str:
        """
        Returns tensor name of last_token_index.
        """
        if self.runner == 'pytorch':
            return PytorchGenerationInputNames.last_token_index.value
        else:
            return self.input_names.last_token_index

    def apply_sliding_window(self):
        """
        Apply a sliding window attention mask (used in Mistral for example). Sliding window attention only attends
        to the last N tokens.
        """
        for bs in range(self.attention_mask.shape[0]):
            sliding_start = max(self.last_token_index[bs, 0] - self.sliding_window + 1, 0)
            if sliding_start > 0:
                self.attention_mask[bs, :sliding_start] = 0

    @counted
    def preprocess_inputs_for_cache_gen(self,
                                        input_ids: torch.Tensor,
                                        attention_mask: torch.Tensor,
                                        static_seq_length: Optional[int] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Preprocess inputs for the cache generation phase, e.g. the first token generation. Because we assume right padding,
        position_ids and attention_mask are fixed and assumed to be generated inside the model.

        For the Samba frontend, the returned input names must be consistent with the input names during tracing (clm_tracer).

        Preprocessings on the inputs include:
        input_ids: zero padding to the static sequence length.

        Args:
            input_ids: Left aligned (right-padded) inputs ids of size (batch_size, input_seq_length).
            # TODO: RK>>   input_seq_length should be smaller than the compiled static sequence length.
            attention_mask: Attention mask of size (batch_size, input_seq_length). Used for calculating lting last_token_index.
            # TODO: RK>> Can't parse "Used for calculating lting last_token_index."
            static_seq_length: The sequence length to pad the input_ids to and can be smaller than max_seq_length.
                               If not specified, it will defaults to max_seq_length.

        Returns:
            Dict of inputs, order insensitive. Does not need to match the model definition since tensors are named.
            Non-tensor inputs are skipped for Samba runner as they are fixed in the PEF during compilation and cannot
            be changed at runtime.

            Returns:
            [input_ids (bs, static_seq_length), last_token_index (bs, 1)].

            Attention mask is not returned because triufill is used instead and position_ids are generated in the model.
            We can do this ONLY because of right padding on input_ids.

            Last token index is passed so the model can slice out the logits corresponding to the token generation
            position. This will save memory bandwidth and also work with Hugging Face's model.generate() function
            even though the input is right padded because the logits would only have sequence length of 1 after slicing.
        """
        expensive_error_if(ValueError, f"input_ids: {input_ids} cannot be left padded", is_left_padded, input_ids,
                           self.pad_token_id)
        expensive_error_if(ValueError,
                           f"attention_mask: {attention_mask} cannot be left padded or contain sliding_window mask",
                           lambda attention_mask: attention_mask is not None and is_left_padded(attention_mask, 0),
                           attention_mask)

        if static_seq_length is None:
            static_seq_length = self.max_seq_length

        self.prompt_tokens = input_ids

        self.input_ids = F.pad(input_ids, (0, static_seq_length - input_ids.shape[1])).int()

        # Use the last 1 in attention mask to indicate the last token. Some tokenizers use a padding token for EOS
        # which makes it impossible to find the last token from right-padded input_ids
        self.last_token_index = torch.argmax(attention_mask.cumsum(1), dim=1, keepdim=True).int()  # shape: [bs, 1]

        self.attention_mask = F.pad(attention_mask, (0, self.max_seq_length - attention_mask.shape[1])).int()

        if self.sliding_window is not None:
            self.apply_sliding_window()

        return {
            self.input_ids_name(token_gen=False, static_seq_length=static_seq_length): self.input_ids,
            self.last_token_index_name: self.last_token_index
        }

    @counted
    def preprocess_inputs_for_token_gen(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Preprocess inputs for the token generation phase. Token generation phase is the calls following cache generation.

        For Samba frontend, the returned input names must be consistent with the inputs names during tracing (clm_tracer).

        Preprocessing on the inputs include:
        input_ids: Slicing out the last non-pad token according to last_token_index so that the token generation model pass
                   always takes input_ids of sequence length of 1.
        attention_mask: Zero padding to the static sequence length and send to model. Attention mask is needed for token
                        generation because RDU does not support non-square triufill.

        Args:
            input_ids: Only the last tokens with sequence length of 1.

        Returns:
            Dict of inputs, order insensitive. Does not need to match the model definition since tensors are named.
            Non-tensors inputs are skipped for Samba runner as they are fixed in the PEF during compilation and cannot
            be changed during runtime.

            Returns:
            [input_ids (bs, 1), attention_mask (bs, static_seq_length), last_token_index (bs, 1)].

            Last token index is passed so the model can slice out the logits corresponding to the token generation
            position. This will save memory bandwidth and also work with Hugging Face's model.generate() function
            even though the input is right padded because the logits would only have a sequence length of 1 after slicing.
        """
        if input_ids.shape[1] > 1:
            raise ValueError(f"input_ids should only be one token in token generation pass")
        self.input_ids = input_ids.int()

        bs = input_ids.shape[0]

        # Cap index within the bound so shorter inputs can keep generating even if longer inputs is full
        self.last_token_index = torch.minimum(
            self.last_token_index + 1,
            torch.full(self.last_token_index.shape, self.max_seq_length - 1).to(input_ids.device))

        # Grow 1 on attention mask
        row_index = torch.arange(0, bs, dtype=int, device=input_ids.device)
        self.attention_mask[row_index, self.last_token_index.squeeze(1)] = 1

        # Add sliding_window mask if needed
        if self.sliding_window is not None:
            self.apply_sliding_window()

        return {
            self.input_ids_name(token_gen=True): self.input_ids,
            self.last_token_index_name: self.last_token_index,
            self.attention_mask_name: self.attention_mask
        }


class CausalLMGenerationMixin:
    """
    A Mixin class to be inherited by SNModelForCausalLM to support Hugging Face's model.generate() function. It provides
    a generic replacement function to replace prepare_inputs_for_generation.
    """
    def __init__(self):
        self.inputs_processor: CachedInferenceRuntime = None
        self.dynamic_input_length: int = None
        self.static_seq_lengths: Set[int] = None
        self.chosen_length: int = None

    def init_inputs_processor(self, runner: Optional[str] = 'pytorch', static_seq_lengths: Optional[Set[int]] = None):
        """
        Initialize the input processor
        Args:
            static_seq_lengths: Static sequence lengths to pad the input_ids to.
            runner: One of ['pytorch', 'samba']. Pytorch runner returns the inputs with names matching the model forward
                    call's function signature (keyword argument). Samba runner returns the inputs with names matching
                    the clm_compile's compile-time tensor symbol name.
        """
        if self.config.pad_token_id is None:
            raise RuntimeError("Please set the pad_token_id into the model config")
        if static_seq_lengths is None:
            static_seq_lengths = set([self.config.max_seq_length])
        elif not isinstance(static_seq_lengths, set):
            raise ValueError(f"static_seq_lengths is expected to be set but got {type(static_seq_lengths)}")
        self.static_seq_lengths = static_seq_lengths

        sliding_window = getattr(self.config, 'sliding_window', None)
        runtime_class = CachedInferenceRuntime.get_registered_runtime(type(self))
        self.inputs_processor = runtime_class(
            self.config.max_seq_length,
            self.config.pad_token_id,
            sliding_window=sliding_window,
            runner=runner,
        )
        self.inputs_processor.preprocess_inputs_for_cache_gen.__dict__['calls'] = 0
        self.inputs_processor.preprocess_inputs_for_token_gen.__dict__['calls'] = 0

    @property
    def runner(self) -> str:
        """
        Returns the runner (samba/pytorch) style.
        """
        return self.inputs_processor.runner

    @runner.setter
    def runner(self, val: str) -> str:
        """
        Sets the runner of the runtime inputs processor
        """
        self.inputs_processor.runner = val

    @property
    def cache_gen(self) -> str:
        """
        Cache generation graph (call schedule) name.
        """
        return self.inputs_processor.cache_gen

    def sn_prepare_inputs_for_generation(self,
                                         input_ids: torch.Tensor,
                                         attention_mask: Optional[torch.Tensor] = None,
                                         use_cache: bool = True,
                                         past_key_values: Optional[torch.Tensor] = None,
                                         **kwargs) -> Dict[str, Any]:
        """
        The function replaces the Hugging Face prepare_inputs_for_generation() function. It is called for
        every token generation. The returned outputs are fed to the model forward pass. The Hugging Face generation
        workflow is the following:

        The Hugging Face's GenerationMixin class does the following:
        1) input_ids and attention_mask initially come as padded tokenized prompts.
        2) input_ids is updated by appending new tokens generated to the tail.
        3) attention_mask is updated by appending 1s to the tail.
        Both input_ids and attention_mask grow in length by 1 in every call
        Hugging Face generation pseudo code:

        While EOS:
            model_inputs = prepare_inputs_for_generation(input_ids, attention_mask, ...)
            logits = model(**model_inputs)

            # -1 is on the sequence dimension, Hugging Face always assumes left padding
            new_token = sample(logits[:, -1, :])

            # Hugging Face always assumes left padding and dynamic input sequence length for generation
            input_ids = input_ids.append(new_token)
            attention_mask = attention_mask.append(1)


        Notice that input_ids as the original inputs is not overwritten by prepare_inputs_for_generation and it keeps
        being appended by new tokens generated.

        This ovewrite ONLY works with right padded tokenization. The outputs are fed directly into Model Zoo model's
        forward pass. An exception will be raised if input_ids is left padded. Because Hugging Face always appends
        the generated token to the input token ids, we have to return last_token_index to feed into the model.

        Args:
            input_ids: Dynamic inputs after prepare_inputs_for_generation, of size [bs, dynamic_input_length]. It's
                       dynamic due to Hugging Face's appending in every token generation.
            attention_mask: Dynamic attention_mask, of size [bs, dynamic_input_length].
                            This input is ONLY used in the first call, e.g. prompt's attention_mask.
                            The following call uses CachedInferenceRuntime's internally updated attention mask because
                            the one from Hugging Face has a dynamic sequence length and cannot be used directly in
                            ModelZoo models.
        Returns:
            Inputs to be fed to the model forward pass.
        """
        self.dynamic_input_length = input_ids.shape[1]
        if not use_cache:
            raise ValueError("Only cached inference is supported")
        if self.inputs_processor is None:
            raise RuntimeError("inputs_processor is uninitialized, call model.init_inputs_processor to initialize")

        cache_gen = self.inputs_processor.preprocess_inputs_for_cache_gen.calls == 0
        if cache_gen:
            # Choose the static_seq_length for the first time call, e.g. cache_gen graph
            self.chosen_length = min([s for s in self.static_seq_lengths if s >= self.dynamic_input_length])
            model_inputs = self.inputs_processor.preprocess_inputs_for_cache_gen(input_ids,
                                                                                 attention_mask=attention_mask,
                                                                                 static_seq_length=self.chosen_length)
        else:
            input_ids = input_ids[:, -1:]
            model_inputs = self.inputs_processor.preprocess_inputs_for_token_gen(input_ids)

        if self.runner == 'pytorch':
            model_inputs['use_cache'] = True
            model_inputs['past_key_values'] = past_key_values if not cache_gen else None
        # Samba does not need non-tensor input like use_cache during runtime. Samba also does not need past_key_values
        # because it's already in RDU memory sharing the same address as the output of the cache generation graph.
        # past_key_values are modified in place in the token generation passes.
        return model_inputs


class PretrainRuntime:
    model_to_runtime: Dict[Type[torch.nn.Module], Type['PretrainRuntime']] = {}

    def __init__(self, max_seq_length: int):
        """
        Base class for preprocessing inputs to run pretraining.

        Args:
            max_seq_length: The static sequence length when the PEF is compiled.
        """
        self.max_seq_length: int = max_seq_length

    def __init_subclass__(cls, model: Type[torch.nn.Module] = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if model is None:
            raise RuntimeError('model needs to be specified to register PretrainRuntime')
        PretrainRuntime.model_to_runtime[model] = cls

    @classmethod
    def get_registered_runtime(cls, model_type: Type[torch.nn.Module]) -> 'PretrainRuntime':
        """
        Args:
            model_type: The class name of the SNModel to be queried for.
        Returns:
            The registered class of the correponding ModelRuntime.
        """
        if model_type not in cls.model_to_runtime:
            raise ValueError(
                f"{model_type} is not registered with PretrainRuntime. Registered are {cls.model_to_runtime.keys()}")
        return cls.model_to_runtime[model_type]

    def pad_tensor_right(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: A tensor that is [bs, dynamic_input_length]
        Returns:
            The input tensor right padded where its new shape is [bs, max_seq_length]
        """
        # NOTE: When padding the input_ids, the padding token id doesn't matter
        # As long as we construct the attn mask and grad loss scale properly
        assert input_tensor.dim() == 2
        return F.pad(input_tensor, (0, self.max_seq_length - input_tensor.shape[1]), "constant",
                     PADDING_TOKEN_TYPE_ID).int()

    def shift_and_pad_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: A tensor that is [bs, dynamic_input_length]
        Returns:
            The input_tensor has the first value of the the second dimension removed,
            the resulting tensor is [bs, dynamic_input_length-1], this is right padded to [bs, max_seq_length]
        """
        assert input_tensor.dim() == 2
        shifted_tensor = input_tensor[..., 1:]
        return self.pad_tensor_right(shifted_tensor)

    def prepare_inputs_to_train(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Dynamic inputs of size [bs, dynamic_input_length].
            token_type_ids: Dynamic token type ids of size [bs, dynamic_input_length].
        Returns:
            input_ids: Padded version of input_ids right padded where its new shape is [bs, max_seq_length].
            attention_mask: The article attention mask represented by two collapsed tensors.  The actual article attention mask
                            is of size [batch_size, 1, sequence_length, sequence_length] generated on RDU.
                            The two collapsed tensors are each of size [batch_size, 1, 1, sequence_length]
            labels: The input_tensor has the first value of the second dimension removed.
                    The resulting tensor is [bs, dynamic_input_length-1], this is right padded to [bs, max_seq_length]
        """
        return {
            'input_ids': self.pad_tensor_right(input_ids),
            'attention_mask': get_collapsed_article_attention_mask(self.shift_and_pad_tensor(token_type_ids)),
            'labels': self.shift_and_pad_tensor(input_ids)
        }

    def get_gradient_scale(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient scale for the loss function.

        Args:
            token_type_ids: Dynamic token type ids of size [bs, dynamic_input_length].
        Returns:
            A tensor where each element (i,j) is
            - 1 if labels[i, j] is a prompt, completion or article sep token (0, 1 or 3)
            - 0 if labels[i, j] is a padding token (2)
        """

        target_token_type_ids = self.shift_and_pad_tensor(token_type_ids)

        # Ignore loss over padding tokens
        grad_scale = ~target_token_type_ids.eq(PADDING_TOKEN_TYPE_ID)

        # Ignore loss for first token in new article
        article_sep_indices = torch.where(target_token_type_ids[:, :-1] == ARTICLE_SEP_TOKEN_TYPE_ID)
        first_tokens_in_article_indices = (article_sep_indices[0], article_sep_indices[1] + 1)
        grad_scale[first_tokens_in_article_indices] = False

        # Normalize the gradient scale here to replace mean reduction
        grad_scale = grad_scale.float()
        grad_scale /= torch.sum(grad_scale)

        return grad_scale.flatten()
