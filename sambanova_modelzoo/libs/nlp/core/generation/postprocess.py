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

from typing import Any, Dict

import torch
from sambanova_modelzoo.libs.nlp.core.generation.sampling import SamplingMethod, SamplingModule
from sambanova_modelzoo.models.custom_ops import scatter

from sambaflow.samba.directives import op_fusion


class PostprocessWithSampling(torch.nn.Module):
    """
    Postprocess to sample out tokens and prepare next iteration's inputs. This includes:
    1. Sampling on logits to token index, including greedy search.
    2. Update attention_mask's last_token_index position to be 1. Attention mask is used by token_gen only.
    3. Update the generated tokens in-place at the position of last_token_index.
    4. Update the generated index in-place

    On RDU dataflow, the load of last_token_index moves data from DDR to SRAM, it happends only once and then the SRAM
    is consumed by increment and attention_mask. The output of increment is written into a different SRAM address and
    then written further back into the same DDR address of the input by region sharing.
    """
    def __init__(self, sampling_method: SamplingMethod, use_plugin_heuristics: bool = False):
        super().__init__()
        self.use_plugin_heuristics = use_plugin_heuristics
        self.sampling_method = sampling_method
        self.sampling_module = SamplingModule.get_registered_module(sampling_method)()

    def forward(self, logits: torch.Tensor, last_token_index: torch.Tensor, attention_mask: torch.Tensor,
                generated_tokens: torch.Tensor, generated_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Logits tensor of token to be generated, (batch_size, 1, vocab_size).
            attention_mask: Attenion mask to be used by token_gen graph only, (batch_size, max_seq_length).
            last_token_index: The index positions of the last non-pad tokens of input_ids, (batch_size, 1).
            generated_tokens: All generated tokens, to be updated with new token at the current position
                              (generated_index + 1), of shape (batch_size, max_seq_length)
            generated_index: The index starting from 0 to put the generated_tokens, (1,).

        Returns:
            Sampled tokens and other in-place updated inputs.
        """
        with op_fusion(func_name=f"postprocess_{self.sampling_method.name}",
                       heuristic={
                           "distribution": "kPostprocess",
                           "tiling": "kPostprocess",
                           "mapping": "kPostprocess"
                       },
                       plugins=["libPostprocessArgmaxHook.so"],
                       user_annotated=self.use_plugin_heuristics):
            batch_size = logits.shape[0]
            max_seq_length = generated_tokens.shape[1]
            assert len(logits.shape
                       ) == 3 and logits.shape[1] == 1, "logits has to be sliced of shape (batch_size, 1, vocab_size)"
            assert len(attention_mask.shape) == 2 and attention_mask.shape[
                0] == batch_size, "token_gen only attention_mask has to be of shape (batch_size, max_seq_length)"
            assert list(last_token_index.shape) == [batch_size,
                                                    1], "last_token_index has to be of shape (batch_size, 1)"
            assert list(generated_index.shape) == [1], "generated_index has to be of shape (1,)"

            tokens = self.sampling_module(logits)
            last_token_index = last_token_index + 1
            # Cap last_token_index here so that shorter prompt can still keep generating even if the longer one in the
            # same batch reached max_seq_length. Capping is important to prevent out-of-bound scatter which has undefined
            # behavior on RDU.
            last_token_index = torch.minimum(last_token_index, torch.full((batch_size, 1), max_seq_length - 1))

            update = torch.ones(batch_size, 1, 1).int()
            start_indices = last_token_index.unsqueeze(1)
            attention_mask = scatter(attention_mask, update, start_indices, scatter_dims=[0], batched_dims=1)

            start_indices = generated_index.expand(batch_size, 1).unsqueeze(1)
            generated_tokens = scatter(generated_tokens,
                                       tokens.reshape(-1, 1, 1),
                                       start_indices,
                                       scatter_dims=[0],
                                       batched_dims=1)
            generated_index = generated_index + 1
            generated_index = torch.minimum(generated_index, torch.tensor([max_seq_length - 1]))

        return {
            'tokens': tokens,
            'last_token_index': last_token_index,
            'attention_mask': attention_mask,
            'generated_tokens': generated_tokens,
            'generated_index': generated_index,
        }


class CausalModelWithSampling(torch.nn.Module):
    """
    A wrapper model that includes ModelForCausalLM and PostprocessWithSampling. This wrapper model expects the 
    lm_model to return hidden states after the last encoder layer (before any normalization) so the remaining graph can
    be fused with the postprocess graph. The remaining graph usually includes norm and lm_head.
    """
    def __init__(self, lm_model: torch.nn.Module, sampling_method: SamplingMethod):
        super().__init__()
        self.lm_model = lm_model
        self.postprocess = PostprocessWithSampling(sampling_method,
                                                   use_plugin_heuristics=lm_model.config.use_plugin_heuristics)

    def forward(self, token_gen_inputs: Dict[str, Any], last_token_index: torch.Tensor, attention_mask: torch.Tensor,
                generated_tokens: torch.Tensor, generated_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        last_token_index and attention_mask are duplicated from the ones in token_gen_inputs because we only want to
        trace the postprocess graph part that's connected to last_token_index and attention_mask, not any nodes from
        token_gen.

        Args:
            token_gen_inputs: Inputs for token generation graph of cached inference.
            attention_mask: Attenion mask to be updated, (batch_size, max_seq_length).
            last_token_index: The index positions of the last non-pad tokens of input_ids, (batch_size, 1).
            generated_tokens: All generated tokens, to be updated with new token at the current position
                              (generated_index + 1), of shape (batch_size, max_seq_length)
            generated_index: The index starting from 0 to put the generated_tokens, (1,).

        Returns:
            Sampled tokens, in-place updated inputs and last_hidden_states for slicing the fused postprocess graph.
        """
        logits, *_, last_hidden_states = self.lm_model(**token_gen_inputs)
        outputs = self.postprocess(logits, last_token_index, attention_mask, generated_tokens, generated_index)
        # last_hidden_states must be appended at the end so tracing outputs can retrieve it from tuple
        outputs['last_hidden_states'] = last_hidden_states
        return outputs
