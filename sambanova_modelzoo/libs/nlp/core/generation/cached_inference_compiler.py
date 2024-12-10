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

from typing import Any, Dict, Optional, Set, Tuple

import torch
from sambanova_modelzoo.libs.nlp.core.clm_runtime import CachedInferenceSambaGraphNames, InferenceSambaGraphNames
from sambanova_modelzoo.libs.nlp.core.clm_tracer import CachedInferenceTracer, SambaGenerationInputNames
from sambanova_modelzoo.libs.nlp.core.generation.postprocess import ModelWithPostprocessInit, PostprocessWithSampling
from sambanova_modelzoo.libs.nlp.core.generation.sampling import SamplingMethod, SamplingModule
from sambanova_modelzoo.libs.nlp.core.generation.token_gen_init import TokenGenInit

import sambaflow.samba as samba
from sambaflow.samba import SambaTensor
from sambaflow.samba.graph import FwdGraph
from sambaflow.samba.utils import trace_multigraph
from sambaflow.samba.utils.pef_utils import get_pefmeta_dict


class BaseCompiler:
    """
    This class abstracts the multigraph stitching for compiling cached inference for a LLM and does not include
    postprocessing (see :class:`CachedInferenceCompiler`).
    """
    def __init__(
            self,
            model: torch.nn.Module,
            batch_size: int,
            static_seq_lengths: Optional[Set[int]] = None,
            token_gen_seq_length: int = 1,
    ):
        """
        A base multigraph compiler to stitch the cache_gen graph and token_gen graph together to run efficient inference.
        All model weights are shared. In addition, KV cache tensors also shared memory between models.
        # TODO: What's KV?
        Args:
            model: The torch model to be compiled.
            batch_size: Static batch size to be compiled.
            static_seq_lengths: Static sequence lengths to pad the input_ids to. The compiler only supports fixed sequence
                                length in the PEF. However, we can support multiple sequence lengths at the same time
                                for cache generation graph. At runtime, the user can choose the graph call of sequence
                                length that fits best with the prompt input length. Doing this can enable faster first
                                token generation.
            token_gen_seq_length: The sequence length for token generation.
        # TODO: RK>> I don't understand "the user can choose the graph call of sequence length". Also, how does the user choose?

        """
        if type(model) not in CachedInferenceTracer.model_to_tracer:
            raise RuntimeError(f"{type(model)} is not registered with CachedInferenceTracer")
        if model.config.tie_word_embeddings:
            raise RuntimeError(
                "tie_word_embeddings must be False, shared weight tensor is not supported in multigraph compiler")

        self.model = model
        if self.model.config.max_seq_length is None:
            raise RuntimeError(
                f"max_seq_length is None, it is a SNPretrainedConfig and needs to be set to define a static graph on RDU"
            )

        if static_seq_lengths is None:
            static_seq_lengths = set([self.model.config.max_seq_length])
        elif not isinstance(static_seq_lengths, set):
            raise ValueError(f"static_seq_lengths is expected to be set but got {type(static_seq_lengths)}")

        self.static_seq_lengths: Set[int] = static_seq_lengths
        self.max_seq_length = self.model.config.max_seq_length

        self.token_gen_seq_length = token_gen_seq_length

        tracer_class = CachedInferenceTracer.get_registered_tracer(type(model))
        self.trace_input_gen = tracer_class(model.config, batch_size, token_gen_seq_length)
        # static_seq_length -> cache_gen inputs dictionary for tracing
        self.cache_gen_inputs: Dict[int, Dict[str, Any]] = {}
        # static_seq_length -> cache_gen traced_outputs mapping
        self.cache_gen_outputs: Dict[int, Tuple[SambaTensor]] = {}

        self.token_gen_inputs: Dict[str, Any] = None
        self.token_gen_outputs: Tuple[SambaTensor] = None

        self.graph_names = InferenceSambaGraphNames()
        self.input_names = SambaGenerationInputNames()

    def to_samba(self):
        """
        Convert torch model to samba model.
        No need to convert sampling module because it's non-parametric.
        """
        samba.from_torch_model_(self.model)

    def trace(self):
        """
        Traces a multigraph for inference to sets up the inputs and outputs. :class:`BaseCompiler` compiles a cache_gen
        graph that generates the first token and the initial KV cache and an optional token_gen graph that computes
        subsequent tokens in the generation.
        """
        self.to_samba()
        for seq_length in self.static_seq_lengths:
            self.cache_gen_inputs[seq_length] = self.trace_input_gen.get_cache_gen_tracing_inputs(seq_length)
            self.cache_gen_outputs[seq_length] = trace_multigraph(self.model,
                                                                  self.cache_gen_inputs[seq_length],
                                                                  trace_prefix=self.graph_names.cache_gen(seq_length))

        # Move add_graphs into tracing so that at runtime, the symbols outside of added graph can be filtered out
        self._add_graphs()

    def _add_graphs(self) -> str:
        """
        Add all graphs to prepare for multigraph compilation.
        """
        for seq_length in self.static_seq_lengths:
            samba.session.add_graph(
                FwdGraph(self.cache_gen_inputs[seq_length],
                         self.cache_gen_outputs[seq_length],
                         name=self.graph_names.cache_gen(seq_length)))

    def compile(self, cfg: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Compile a multigraph PEF to do cached inference
        Args:
            cfg: Compilation configuration.
            kwargs: Additional keyword compilation arguments to overwrite the cfg.
        Returns:
            Compiled PEF file's full path name.
        """
        self.trace()

        # metadata used by Samba 1 Turbo. 'lengths' should be a list instead of a set, otherwise it will be parsed in
        # as a string when the metadata is loaded from the PEF
        extra_metadata = {
            "batch_size": self.trace_input_gen.batch_size,
            "max_pef_length": self.max_seq_length,
            "lengths": list(self.static_seq_lengths)
        }

        metadata_dict = {} if cfg is None else get_pefmeta_dict({**cfg, **extra_metadata}, self.model)

        return samba.session.compile_multigraph(name='transformers_textgen_multigraph',
                                                config_dict=cfg if cfg is not None else {},
                                                pef_metadata=metadata_dict,
                                                enable_hypersection=True,
                                                inference=True,
                                                **kwargs)


class CachedInferenceCompiler(BaseCompiler):
    def __init__(self,
                 model: torch.nn.Module,
                 batch_size: int,
                 static_seq_lengths: Optional[Set[int]] = None,
                 token_gen_seq_length: int = 1):
        """
        The compiler class with postprocess modules for all sampling modules. Also, the PostprocessInit modules are
        combined with the cache_gen graphs.
        Args:
            model: The torch model to be compiled.
            batch_size: Static batch size to be compiled.
            static_seq_lengths: Static sequence lengths to pad the input_ids to. The compiler only supports fixed sequence
                                length in the PEF. However, we can support multiple sequence lengths at the same time
                                for cache generation graph. At runtime, the user can choose the graph call of sequence
                                length that fits best with the prompt input length. Doing this can enable faster first
                                token generation.
            token_gen_seq_length: The sequence length for token generation.
        """
        super().__init__(model,
                         batch_size,
                         static_seq_lengths=static_seq_lengths,
                         token_gen_seq_length=token_gen_seq_length)

        self.graph_names = CachedInferenceSambaGraphNames()
        # Token Gen Init Graph
        self.token_gen_init = TokenGenInit(model.config.max_seq_length, model.config.use_plugin_heuristics,
                                           model.config.run_early_tp)
        self.token_gen_init_inputs: Dict[str, Any] = {}
        self.token_gen_init_outputs: Tuple[SambaTensor] = None

        self.postprocess_inputs: Dict[SamplingMethod, Dict[str, SambaTensor]] = {}
        self.postprocess_outputs: Dict[SamplingMethod, Tuple[SambaTensor, ...]] = {}
        # Postprocessing with sampling modules
        self.postprocess_modules: Dict[SamplingMethod, PostprocessWithSampling] = {
            s: PostprocessWithSampling(s, model.config.use_plugin_heuristics, model.config.run_early_tp)
            for s in SamplingMethod if SamplingModule.get_registered_module(s).is_ready_for_deployment()
        }

        # The wrapper module that contains the model (cache gen graph) and postprocess_init module, which is used to
        # create the token_count tensor used for repetition penalty
        self.model_with_postprocess_init = {
            seq_length: ModelWithPostprocessInit(self.model, batch_size, model.config.vocab_size, seq_length)
            for seq_length in self.static_seq_lengths
        }

    def to_samba(self):
        super().to_samba()
        for m in self.postprocess_modules.values():
            samba.from_torch_model_(m)
        samba.from_torch_model_(self.token_gen_init)

    def trace(self):
        """
        Trace a multigraph for cached inference to set up the inputs and outputs. The graph tracing folows the following
        dependency:
        cache_gen -> sampling -> token_gen -> sampling
                                    ^           |
                                    |------------
        """
        self.to_samba()
        for seq_length in self.static_seq_lengths:
            self.cache_gen_inputs[seq_length] = self.trace_input_gen.get_cache_gen_tracing_inputs(seq_length)
            self.cache_gen_outputs[seq_length] = trace_multigraph(
                self.model_with_postprocess_init[seq_length], {'cache_gen_inputs': self.cache_gen_inputs[seq_length]},
                trace_prefix=self.graph_names.cache_gen(seq_length))

        # Use any cache_gen outputs to get KV cache input to trace token_gen graph. Later on, we will stitch all KV
        # cache outputs to share the same region.
        self.token_gen_inputs = self.trace_input_gen.get_token_gen_tracing_inputs(self.cache_gen_outputs[next(
            iter(self.static_seq_lengths))])

        self.token_gen_outputs = trace_multigraph(self.model,
                                                  self.token_gen_inputs,
                                                  trace_prefix=self.graph_names.token_gen(self.max_seq_length))

        self._trace_for_postprocess_graph()

        # TokenGenInitGraph not supported in draft model
        if self.token_gen_seq_length == 1:
            self._trace_for_token_gen_init_graph()
            self._connect_token_gen_init_graph()

        self._connect_kv_cache_from_cache_gen_to_token_gen()
        self._connect_kv_cache_token_gen_output_to_token_gen_input()
        self._connect_logits()
        self._connect_postprocess_graphs()
        # Move add_graphs into tracing so that at runtime, the symbols outside of added graph can be filtered out
        self._add_graphs()

    def _connect_kv_cache_from_cache_gen_to_token_gen(self):
        """
        Using shared sn_region_name to connect between inputs and outputs to share the same memory address.
        We already used one of the cache_gen's KV cache outputs to trace token_gen graph. Now we need to
        1. Connect all cache_gen's KV cache output to share the region with token_gen's KV cache input.
        2. Connect KV cache output of token_gen to its input so it becomes an in-place update.
        """
        # Connect all kv cache outputs
        for cache_gen_outputs in self.cache_gen_outputs.values():
            cache_gen_kvs = self.trace_input_gen._get_kv_cache(cache_gen_outputs)
            token_gen_kvs = self.token_gen_inputs['past_key_values']
            for (cache_gen_k, cache_gen_v), (token_gen_k, token_gen_v) in zip(cache_gen_kvs, token_gen_kvs):
                cache_gen_k.sn_region_name = token_gen_k.sn_region_name
                cache_gen_v.sn_region_name = token_gen_v.sn_region_name

    def _connect_kv_cache_token_gen_output_to_token_gen_input(self):
        """
        Update the region name of the KV cache output of the token_gen graph to be the same as the input KV cache.
        The model currently implements out-of-place ops such as scatter to update the cache, for example:

        layer_past_key = scatter(layer_past_key, key_states, last_token_index, [0, 2])

        Without updating the output layer_past_key region_name, the compiler recognizes this scatter operation
        as an out-of-place operation and return a new tensor. With Samba frontend, at runtime, we have to
        transfer the output KV cache from the device to the host and then feed it back to the device as input.
        That incurs a host to device memory transfer, which significantly impacts performance.
        This limitation will be lifted with a future frontend. Currently, the workaround is
        to set the region name to be the same between input cache and output cache so that this operation
        becomes an inplace update. Thus reducing the costly device to host or host to device memory transfer.
        """
        output_cache = self.trace_input_gen._get_kv_cache(self.token_gen_outputs)
        input_cache = self.token_gen_inputs['past_key_values']
        for ipt, out in zip(input_cache, output_cache):
            # For cached inference graph, we already assign token_gen graph's input cache region name by the cache_gen's
            # output region name. So it's CRITICAL that we assign the updated input cache region name to the output cache
            # for the token_gen graph. ipt.sn_region_name = out[0].sn_region_name will NOT work.
            out[0].sn_region_name = ipt[0].sn_region_name
            out[1].sn_region_name = ipt[1].sn_region_name

    def _connect_logits(self) -> None:
        """Uses shared sn_region_name to connect the cache_gen output logit tensor and the input logit tensor to share
        the same memory address."""
        for cache_gen_outputs in self.cache_gen_outputs.values():
            logits = self.trace_input_gen.get_logits_output(cache_gen_outputs)
            logits.sn_region_name = self.input_names.logits
        self.trace_input_gen.get_logits_output(self.token_gen_outputs).sn_region_name = self.input_names.logits

    def _trace_for_token_gen_init_graph(self):
        self.token_gen_init_inputs = self.trace_input_gen.get_token_gen_init_tracing_inputs()
        self.token_gen_init_outputs = trace_multigraph(self.token_gen_init,
                                                       self.token_gen_init_inputs,
                                                       trace_prefix=self.graph_names.token_gen_init())

    def _connect_token_gen_init_graph(self):
        attention_mask = self.trace_input_gen.get_attention_mask_token_gen_init_output(self.token_gen_init_outputs)
        generated_index = self.trace_input_gen.get_generated_index_token_gen_init_output(self.token_gen_init_outputs)
        attention_mask.sn_region_name = self.input_names.attention_mask
        generated_index.sn_region_name = self.input_names.generated_index

    def _add_token_gen_init_graph(self):
        """ Add standalone token gen init graph """
        samba.session.add_graph(
            FwdGraph(self.token_gen_init_inputs, self.token_gen_init_outputs, name=self.graph_names.token_gen_init()))

    def _connect_postprocess_graphs(self):
        self._connect_postprocess_inplace_updates(self.postprocess_outputs)

        for cache_gen_outputs in self.cache_gen_outputs.values():
            token_count = self.trace_input_gen.get_token_count_output(
                model_with_postprocess_init_outputs=cache_gen_outputs)
            token_count.sn_region_name = self.input_names.token_count

    def _add_postprocess_graphs(self):
        """ Add standalone postprocess graphs """
        for sampling_method, outputs in self.postprocess_outputs.items():
            samba.session.add_graph(
                FwdGraph(self.postprocess_inputs[sampling_method],
                         outputs,
                         name=self.graph_names.postprocess(sampling_method)))

    def _add_graphs(self) -> str:
        """
        Add all graphs to prepare for multigraph compilation.
        """
        # add cache_gen graph
        super()._add_graphs()

        # add token_gen graph
        samba.session.add_graph(
            FwdGraph(self.token_gen_inputs,
                     self.token_gen_outputs,
                     name=self.graph_names.token_gen(self.max_seq_length)))

        self._add_token_gen_init_graph()
        self._add_postprocess_graphs()

    def _trace_for_postprocess_graph(self):
        """ Standalone postprocessing with sampling graph """
        for sampling_method, module in self.postprocess_modules.items():
            inputs = self.trace_input_gen.get_postprocess_tracing_inputs(sampling_method)
            outputs = trace_multigraph(module, inputs, trace_prefix=self.graph_names.postprocess(sampling_method))
            self.postprocess_inputs[sampling_method] = inputs
            self.postprocess_outputs[sampling_method] = outputs

    def _connect_postprocess_inplace_updates(self, postprocess_outputs: Dict[SamplingMethod, Tuple[SambaTensor, ...]]):
        """
        Connects postprocess graph inputs and outputs so eligible tensors can share the same memory address. All
        postprocess graphs use the same input tensors for tracing so they all share the same input region.
        Specifically, this function:
        Connects last_token_index input and output to enable in-place updates.
        Connects attention_mask input and output to enable in-place updates.
        Connects the token output back to token_gen graph's input_ids to avoid device <-> host transfer.
        Connects the generated_tokens input and output to enable in-place update.
        Connects the generated_tokens_streaming input and output to enable in-place update.
        Connects the generated_index input and output to enable in-place update.
        Connects the token_count input and output to enable in-place update.
        """
        for tokens, last_token_index, attention_mask, generated_tokens, generated_tokens_streaming, generated_index, *args in postprocess_outputs.values(
        ):
            # Connect sampling output to token_gen input_ids
            tokens.sn_region_name = self.token_gen_inputs['input_ids'].sn_region_name
            # In-place updates
            last_token_index.sn_region_name = self.input_names.last_token_index
            attention_mask.sn_region_name = self.input_names.attention_mask
            generated_tokens.sn_region_name = self.input_names.generated_tokens
            generated_tokens_streaming.sn_region_name = self.input_names.generated_tokens_streaming
            generated_index.sn_region_name = self.input_names.generated_index
            # Speculative decoding does not support repetition penalty
            if last_token_index.shape[-1] == 1:
                args[0].sn_region_name = self.input_names.token_count
