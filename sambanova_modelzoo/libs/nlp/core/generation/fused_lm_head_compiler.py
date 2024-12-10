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

from typing import Dict, Optional, Set, Tuple

import torch
from sambanova_modelzoo.libs.nlp.core.generation.postprocess import CausalModelWithSampling, PostprocessWithSampling
from sambanova_modelzoo.libs.nlp.core.generation.sampling import SamplingMethod, SamplingModule

import sambaflow.samba as samba
from sambaflow.samba import SambaTensor
from sambaflow.samba.graph import FwdGraph
from sambaflow.samba.utils import trace_multigraph

from .cached_inference_compiler import CachedInferenceCompiler

# TODO: An intro to what this file does would be great. Specifically, explain what lm_head does!


class FusedLMHeadCompiler(CachedInferenceCompiler):
    def __init__(
            self,
            model: torch.nn.Module,
            batch_size: int,
            static_seq_lengths: Optional[Set[int]] = None,
    ):
        """
        A multigraph compiler to stitch the cache_gen graph and token_gen graphs to run efficient inference. All model
        weights are shared. In addition, KV cache tensors also shared memory between models.
        This class differs from CachedInferenceCompiler. Here, the lm_head is already in its standalone graph schedule.
        # TODO: Why schedule?
        In addition, each sampling module can decide whether to fuse with lm_head for performance.

        Args:
            model: The main LM torch model to be compiled and shared between different graphs.
            batch_size: Static batch size to be compiled.
            static_seq_lengths: Static sequence lengths to pad the input_ids to. The compiler only supports fixed sequence
                                length in the PEF. However, we can support multiple sequence lengths at the same time
                                for the cache generation graph. At runtime, users can choose the graph call of sequence
                                length that fits best with the prompt input length. Doing this can enable faster first
                                token generation.
        """
        super().__init__(model, batch_size, static_seq_lengths=static_seq_lengths)
        self.model = model

        self.lm_model_with_sampling_inputs: Dict[SamplingMethod, Dict[str, SambaTensor]] = {}
        self.lm_model_with_sampling_outputs: Dict[SamplingMethod, Tuple[SambaTensor, ...]] = {}
        self.fused_postprocess_outputs: Dict[SamplingMethod, Tuple[SambaTensor, ...]] = {}

        # Wrapper modules with main model and postprocessing to fuse lm_head with sampling.
        self.lm_model_with_sampling_modules: Dict[SamplingMethod, CausalModelWithSampling] = {}

        # Use each SamplingModule either with a standalone graph schedule or fused with lm_head for performance
        # TODO: Is the change to the comment above correct? What's lm_head?
        for s in SamplingMethod:
            if SamplingModule.get_registered_module(s).is_ready_for_deployment():
                if SamplingModule.get_registered_module(s).is_fused_with_lm_head():
                    self.lm_model_with_sampling_modules[s] = CausalModelWithSampling(model, s)
                else:
                    self.postprocess_modules[s] = PostprocessWithSampling(s, model.config.use_plugin_heuristics,
                                                                          model.config.run_early_tp)

    def _trace_for_postprocess_graph(self):
        """ Called after tracing main graphs """
        # Standalone postprocessing with sampling
        super()._trace_for_postprocess_graph()

        # Postprocess to do sampling fused with lm_head
        # TODO: what's lm_head?
        with self.model.hyperfunction.fuse_lm_head_with_sampling():
            for sampling_method, module in self.lm_model_with_sampling_modules.items():
                inputs = {
                    "token_gen_inputs": self.token_gen_inputs,
                    "last_token_index": self.trace_input_gen.get_last_token_index(),
                    "attention_mask": self.trace_input_gen.get_attention_mask(),
                    "generated_tokens": self.trace_input_gen.get_generated_tokens(),
                    "generated_tokens_streaming": self.trace_input_gen.get_generated_tokens_streaming(),
                    "generated_index": self.trace_input_gen.get_generated_index(),
                    "repetition_penalty": self.trace_input_gen.get_repetition_penalty(),
                    'token_count': self.trace_input_gen.get_token_count(),
                }
                if sampling_method == SamplingMethod.multinomial:
                    inputs.update({
                        "temperature": self.trace_input_gen.get_temperature(),
                        "top_k": self.trace_input_gen.get_top_k(),
                        "top_p": self.trace_input_gen.get_top_p(),
                        "pre_generated_randoms": self.trace_input_gen.get_pre_generated_randoms()
                    })
                outputs = trace_multigraph(module, inputs, trace_prefix=self.graph_names.postprocess(sampling_method))
                self.lm_model_with_sampling_inputs[sampling_method] = inputs
                self.lm_model_with_sampling_outputs[sampling_method] = outputs
                # discard the last_hidden_states to be used for inplace postprocess connections
                self.fused_postprocess_outputs[sampling_method] = outputs[:-1]

    def _connect_postprocess_graphs(self):
        """
        Connects postprocessing graph inputs and outputs so eligible tensors share the same memory address. All postprocessing
        graphs use the same input tensors for tracing so they all share the same input region.

        Specifically, this function:
        Connects the postprocessing graph inputs and outputs with the hidden_states of the cache_gen and token_gen graphs
        by sharing the sn_region_name.
        """
        # postprocessing inplace updates
        super()._connect_postprocess_graphs()

        # postprocessing inplace updates with fused lm_head
        self._connect_postprocess_inplace_updates(self.fused_postprocess_outputs)

        # cache_gen
        for cache_gen_outputs in self.cache_gen_outputs.values():
            hidden_states = self.trace_input_gen.get_last_hidden_states_output(cache_gen_outputs=cache_gen_outputs)
            hidden_states.sn_region_name = self.input_names.last_hidden_states
        # token_gen
        hidden_states = self.trace_input_gen.get_last_hidden_states_output(token_gen_outputs=self.token_gen_outputs)
        hidden_states.sn_region_name = self.input_names.last_hidden_states

        # postprocessing fused with lm_head
        for outputs in self.lm_model_with_sampling_outputs.values():
            outputs[-1].sn_region_name = self.input_names.last_hidden_states

    def to_samba(self):
        """
        Convert torch model to Samba model.
        No need to convert sampling module because it's non-parametric.
        # TODO: Does everyone know what non-parametric means in this context?
        """
        super().to_samba()
        for m in self.lm_model_with_sampling_modules.values():
            samba.from_torch_model_(m)

    @staticmethod
    def _swap(old: SambaTensor, new: SambaTensor):
        """ Swap old tensor with new tensor """
        new.dsts = new.dsts + old.dsts
        for dst in old.dsts:
            for i, ipt in enumerate(dst.inputs):
                if id(ipt) == id(old):
                    dst.inputs[i] = new

    def _add_graphs(self) -> str:
        """
        Add all graphs to prepare for multigraph compilation.
        """
        # cache gen graphs without lm_head
        for seq_length in self.static_seq_lengths:
            cache_gen_outputs = self.cache_gen_outputs[seq_length]
            last_hidden_states = self.trace_input_gen.get_last_hidden_states_output(cache_gen_outputs=cache_gen_outputs)
            kvs = self.trace_input_gen._get_kv_cache(cache_gen_outputs)
            samba.session.add_graph(
                FwdGraph(self.cache_gen_inputs[seq_length], (last_hidden_states, kvs),
                         name=self.graph_names.cache_gen(seq_length)))

        # token gen graph without lm_head
        last_hidden_states = self.trace_input_gen.get_last_hidden_states_output(
            token_gen_outputs=self.token_gen_outputs)
        kvs = self.trace_input_gen._get_kv_cache(self.token_gen_outputs)
        samba.session.add_graph(
            FwdGraph(self.token_gen_inputs, (last_hidden_states, kvs),
                     name=self.graph_names.token_gen(self.max_seq_length)))

        # standalone lm_head graph for sampling on CPU
        logits_output = self.trace_input_gen.get_logits_output(self.token_gen_outputs)
        samba.session.add_graph(FwdGraph((last_hidden_states, ), (logits_output, ), name=self.graph_names.lm_head()))

        # postprocessing graphs fused with lm_head
        for sampling_method, outputs in self.lm_model_with_sampling_outputs.items():
            # MAC does not allow different inputs share region so we use a unified last_hidden_states here
            self._swap(outputs[-1], last_hidden_states)
            fused_postprocess_inputs = (
                last_hidden_states,  # starts from last_hidden_states to include norm and lm_head
                *self.lm_model_with_sampling_inputs[sampling_method].values()),

            samba.session.add_graph(
                FwdGraph(fused_postprocess_inputs,
                         self.fused_postprocess_outputs[sampling_method],
                         name=self.graph_names.postprocess(sampling_method)))
