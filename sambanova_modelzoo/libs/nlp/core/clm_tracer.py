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

from typing import Any, Dict, List, Tuple, Type

import torch
from transformers import PretrainedConfig

from sambaflow import samba
from sambaflow.samba import SambaTensor

from sambanova_modelzoo.libs.nlp.core.clm_runtime import SambaGenerationInputNames, SambaPretrainInputNames


class CachedInferenceTracer:
    model_to_tracer: Dict[Type[torch.nn.Module], Type['CachedInferenceTracer']] = {}

    def __init__(self, config: PretrainedConfig, batch_size: int):
        """
        Base class for generating dummy inputs for Samba tracing on cached inference generative task

        Args:
            config: Model config object
            batch_size: Batch size since we only support static tracing
        """
        self.config = config
        self.batch_size = batch_size
        self.input_names = SambaGenerationInputNames()
        self._last_token_index: SambaTensor = None
        self._attention_mask: SambaTensor = None
        self._logits: SambaTensor = None
        self._cache_gen_input_ids: Dict[int, SambaTensor] = {}
        self._token_gen_input_ids: SambaTensor = None
        self._generated_tokens: SambaTensor = None
        self._generated_index: SambaTensor = None

    def __init_subclass__(cls, model: Type[torch.nn.Module] = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if model is None:
            raise RuntimeError('model needs to be specified to register CachedInferenceTracer')
        CachedInferenceTracer.model_to_tracer[model] = cls

    @classmethod
    def get_registered_tracer(cls, model_type: Type[torch.nn.Module]) -> "CachedInferenceTracer":
        if model_type not in cls.model_to_tracer:
            raise ValueError(
                f"{model_type} is not registered with CachedInferenceTracer. Registered are {cls.model_to_tracer.keys()}"
            )
        return cls.model_to_tracer[model_type]

    @property
    def num_key_value_heads(self) -> int:
        """
        Returns the number of heads for generating key and value matrices
        """
        return self.config.num_attention_heads

    def get_cache_gen_input_ids(self, static_seq_length: int) -> SambaTensor:
        """
        Args:
            static_seq_length: The sequence length of padded inputs.
        Returns:
            The dummy input token ids to trace cache_gen graph.
        """
        if static_seq_length in self._cache_gen_input_ids:
            return self._cache_gen_input_ids
        input_ids = torch.randint(0, 5000, (self.batch_size, static_seq_length)).int()
        input_ids = samba.from_torch_tensor(input_ids, name=self.input_names.cache_gen_input_ids(static_seq_length))
        self._cache_gen_input_ids[static_seq_length] = input_ids
        return input_ids

    def get_token_gen_input_ids(self) -> SambaTensor:
        """
        Returns the dummy input token ids to trace cache_gen graph
        """
        if self._token_gen_input_ids is not None:
            return self._token_gen_input_ids
        input_ids = torch.randint(0, 5000, (self.batch_size, 1)).int()
        input_ids = samba.from_torch_tensor(input_ids,
                                            name=self.input_names.token_gen_input_ids(self.config.max_seq_length))
        self._token_gen_input_ids = input_ids
        return input_ids

    def get_attention_mask(self) -> SambaTensor:
        """
        Returns the dummy attention mask of shape (batch_size, max_seq_length) for token_gen graph and postprocess graph.
        """
        if self._attention_mask is not None:
            return self._attention_mask
        attention_mask = torch.randint(2, (self.batch_size, self.config.max_seq_length), dtype=torch.int)
        attention_mask = samba.from_torch_tensor(attention_mask, name=self.input_names.attention_mask)
        self._attention_mask = attention_mask
        return attention_mask

    def get_logits(self) -> SambaTensor:
        """
        Returns the logits of shape (batch_size, 1, vocab_size) to trace the postprocess graphs.
        """
        if self._logits is not None:
            return self._logits
        logits = torch.rand([self.batch_size, 1, self.config.vocab_size], dtype=torch.float)
        logits = samba.from_torch_tensor(logits, name=self.input_names.logits)
        self._logits = logits
        return logits

    def get_generated_tokens(self) -> SambaTensor:
        """
        Returns the generated_tokens to be updated and used as cache to hold new tokens.
        generated_tokens is of size (batch_size, max_seq_length)
        """
        if self._generated_tokens is not None:
            return self._generated_tokens
        generated_tokens = torch.randint(0, 5000, (self.batch_size, self.config.max_seq_length)).int()
        generated_tokens = samba.from_torch_tensor(generated_tokens, name=self.input_names.generated_tokens)
        self._generated_tokens = generated_tokens
        return generated_tokens

    def get_generated_index(self) -> SambaTensor:
        """
        Returns the generated_index to update generated_tokens. It starts from 0 and increment in postprocess graph
        by 1 in each call.
        """
        if self._generated_index is not None:
            return self._generated_index
        generated_index = torch.zeros(1, dtype=torch.int32)
        generated_index = samba.from_torch_tensor(generated_index, name=self.input_names.generated_index)
        self._generated_index = generated_index
        return generated_index

    def get_postprocess_tracing_inputs(self) -> Dict[str, SambaTensor]:
        """
        Returns the logits, last non-pad token index (last_token_index) for each sequence and attention_mask of token_gen
        graph to be postprocessed.

        Returns:
            logits, last_token_index and attention_mask for postprocessing
        """
        return {
            'logits': self.get_logits(),
            'last_token_index': self.get_last_token_index(),
            'attention_mask': self.get_attention_mask(),
            'generated_tokens': self.get_generated_tokens(),
            'generated_index': self.get_generated_index()
        }

    def get_last_token_index(self) -> SambaTensor:
        """
        Returns the last non-pad token index for each sequence. It can be used for:
        1. placing the current key/value to the actual position of the static key/value cache of full sequence length.
        2. slicing the actual logits to be used for generating the next token.
        Only one last_token_index can be used to trace multigraph. We allow one input to share region name with multiple
        outputs. However we don't allow multiple inputs share same region name. last_token_index is used as inputs to
        cache_gen, token_gen and postprocess graph and share the same address so it has to be one tensor instance.

        Returns:
            Last token indices SambaTensor
        """
        if self._last_token_index is not None:
            return self._last_token_index
        last_indices = torch.zeros(self.batch_size, 1, dtype=torch.int32)
        last_indices = samba.from_torch_tensor(last_indices, name=self.input_names.last_token_index)
        self._last_token_index = last_indices
        return last_indices

    def get_cache_gen_tracing_inputs(self, static_seq_length: int) -> Dict[str, Any]:
        """
        Get input tensors to trace the model for cache gen graph
        Args:
            static_seq_length: The sequence length of padded inputs.

        Returns:
            Dict of inputs, with the key matching the model forward function signature
        """
        return {
            'input_ids': self.get_cache_gen_input_ids(static_seq_length),
            'use_cache': True,
            'return_dict': False,
            'last_token_index': self.get_last_token_index(),
        }

    def get_token_gen_tracing_inputs(self, cache_gen_outputs: Tuple[SambaTensor]) -> Dict[str, Any]:
        """
        Get input tensors to trace the model for token gen graph

        Returns:
            Dict of inputs, with the key matching the model forward function signature
        """
        return {
            'input_ids': self.get_token_gen_input_ids(),
            'attention_mask': self.get_attention_mask(),
            'past_key_values': self.get_past_kv_cache(cache_gen_outputs),
            'use_cache': True,
            'return_dict': False,
            'last_token_index': self.get_last_token_index(),
        }

    def _get_kv_cache(self, outputs: Tuple[Any, ...]) -> List[Tuple[SambaTensor, SambaTensor]]:
        """ Returns the list of KV cache for all layers from the traced output """
        raise NotImplementedError

    def get_logits_output(self, outputs: Tuple[SambaTensor, ...]) -> SambaTensor:
        """
        Returns the logits output from all model outputs. By default, returns the first tensor.
        """
        return outputs[0]

    def get_last_hidden_states_output(self, outputs: Tuple[SambaTensor, ...]) -> SambaTensor:
        """
        Returns the last hidden states output (before normalizatiion) from all model outputs. 
        By default, returns the last tensor.
        """
        return outputs[-1]

    def get_past_kv_cache(self, cache_gen_outputs: Tuple[Any, ...]) -> List[Tuple[SambaTensor, SambaTensor]]:
        """
        Returns the list of KV cache for all layers to trace the token generation model, the device memory are shared
        between cache_gen_kv and token_gen_kv to avoid unnecessary memory copy
        """
        kvs = self._get_kv_cache(cache_gen_outputs)
        kv_inputs = []
        if len(kvs) != self.config.num_hidden_layers:
            raise RuntimeError(
                f"kv cache if of length {len(kvs)} does not equals the number of hidden layers {self.config.num_hidden_layers}"
            )
        for k, v in kvs:
            # Specifying the same region_name to share the device memory during compilation
            k_input = SambaTensor(name=k.sn_name + "_copy", region_name=k.sn_region_name, shape=k.shape, dtype=k.dtype)
            v_input = SambaTensor(name=v.sn_name + "_copy", region_name=v.sn_region_name, shape=v.shape, dtype=v.dtype)
            kv_inputs.append((k_input, v_input))
        return kv_inputs


class PretrainTracer:
    model_to_tracer: Dict[Type[torch.nn.Module], Type['PretrainTracer']] = {}

    def __init__(self, config: PretrainedConfig, batch_size: int):
        """
        Base class for generating dummy inputs for Samba tracing on training task

        Args:
            config: Model config object
            batch_size: Batch size since we only support static tracing
        """
        self.config = config
        self.batch_size = batch_size
        self.samba_name = SambaPretrainInputNames()

    @classmethod
    def get_registered_tracer(cls, model_type: Type[torch.nn.Module]) -> "PretrainTracer":
        if model_type not in cls.model_to_tracer:
            raise ValueError(
                f"{model_type} is not registered with PretrainTracer. Registered are {cls.model_to_tracer.keys()}")
        return cls.model_to_tracer[model_type]

    def __init_subclass__(cls, model: Type[torch.nn.Module] = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if model is None:
            raise RuntimeError('model needs to be specified to register PretrainTracer')
        PretrainTracer.model_to_tracer[model] = cls

    def get_input_ids(self) -> SambaTensor:
        """
        Returns:
            The dummy input token ids if input_ids is not provided, otherwise converts input_ids to SambaTensor
        """
        input_ids = torch.randint(0, 5000, (self.batch_size, self.config.max_seq_length)).int()
        input_ids = samba.from_torch_tensor(input_ids, name=self.samba_name.input_ids)
        return input_ids

    def get_article_attention_mask(self) -> Tuple[SambaTensor, SambaTensor]:
        """
        Prepare the article attention mask used for tracing which needs to be sent to device from host during runtime.
        Article attention constrains the attention to only attend within it's article.

        Returns:
            The a pair of vectors for on-chip causal mask generation for article attention
        """
        name_pair = self.samba_name.attention_mask
        # For training we always provides a pair of vector to generate on-chip 3D attention mask
        attention_mask_collapsed_1 = torch.ones((self.batch_size, 1, 1, self.config.max_seq_length))
        attention_mask_collapsed_1 = samba.from_torch_tensor(attention_mask_collapsed_1, batch_dim=0, name=name_pair[0])
        attention_mask_collapsed_2 = torch.ones((self.batch_size, 1, self.config.max_seq_length, 1))
        attention_mask_collapsed_2 = samba.from_torch_tensor(attention_mask_collapsed_2, batch_dim=0, name=name_pair[1])
        return attention_mask_collapsed_1, attention_mask_collapsed_2

    def get_labels(self) -> SambaTensor:
        """
        Returns:
            The label tensor for training.
        """
        labels = torch.ones(self.batch_size, self.config.max_seq_length, dtype=torch.int32)
        return samba.from_torch_tensor(labels, name=self.samba_name.labels)

    def get_tracing_inputs(self) -> Dict[str, SambaTensor]:
        """
        Returns input tensors to trace the model for pretrain.

        Returns:
            Dictionary of inputs for pretrain tracing.
        """
        return {
            'input_ids': self.get_input_ids(),
            'attention_mask': self.get_article_attention_mask(),
            'labels': self.get_labels()
        }
