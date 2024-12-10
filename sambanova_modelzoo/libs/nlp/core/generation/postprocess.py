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

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
from sambanova_modelzoo.libs.nlp.core.generation.sampling import SamplingMethod, SamplingModule
from sambanova_modelzoo.models.custom_ops import scatter, sn_imm, sn_select, sn_zipmapreduce
from sambanova_modelzoo.models.directives import opfusion_id

from sambaflow.samba.directives import op_fusion
from sambaflow.samba.sambatensor import from_torch_tensor


class PostprocessInit(torch.nn.Module):
    """Initializes the token_count tensor that is needed to compute the repetition penalty during sampling.
    Args:
        batch_size: the batch size
        vocab_size: the vocabulary size
    """
    def __init__(self, batch_size: int, vocab_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, last_token_index: torch.Tensor):
        """Computes the token_count tensor, which is a (batch_size, vocab_size) tensor that is a count of how often each
        token has been encountered, both in the input prompt and in the generated tokens.
        Args:
            input_ids: the tokens in the input prompt
            last_token_index: indicates the index of the last valid token for each sample in generated_tokens, of shape
                (batch_size, 1)

        Returns:
            the token_count tensor
        """
        if self.batch_size != input_ids.shape[0]:
            raise ValueError(
                f"Batch size of input_ids {input_ids.shape[0]} does not match self.batch_size {self.batch_size}.")
        if self.batch_size != last_token_index.shape[0]:
            raise ValueError(
                f"Batch size of last_token_index {attention_mask.shape[0]} does not match self.batch_size {self.batch_size}."
            )

        sequence_length = input_ids.shape[1]

        # reconstruct attention_mask from last_token_index

        temp = (torch.arange(sequence_length).repeat(self.batch_size, 1) - last_token_index).to(torch.int32)
        attention_mask = sn_zipmapreduce(
            lambda attrs, temp: sn_select(temp < sn_imm(0, dtype=torch.int32), sn_imm(1, dtype=torch.int32),
                                          sn_imm(0, dtype=torch.int32)), [temp])

        # identifies the tokens in input prompt to penalize, filters out padding tokens using the attention mask.
        # used_tokens will be used as indices, so having a value `vocab_size` will be out-of-bounds and will be
        # a no-op for sn_scatter
        used_tokens = sn_zipmapreduce(lambda attrs, input_ids, attention_mask: sn_select(
            attention_mask > sn_imm(0, dtype=torch.int32), input_ids, sn_imm(self.vocab_size, dtype=torch.int32)),
                                      [input_ids, attention_mask])  # nonpytorch behavior not-yet-generated tokens

        mask_value = torch.ones([self.batch_size, sequence_length, 1], dtype=torch.int32)

        token_count = scatter(
            operand=torch.zeros((self.batch_size, self.vocab_size), dtype=torch.int32),
            update=mask_value,  # BS, SS, 1
            start_indices=used_tokens.unsqueeze(-1),  # input_ids.shape, 1
            scatter_dims=[0],
            rmw_op="kAdd",
            batched_dims=1)

        return token_count


class PostprocessWithSampling(torch.nn.Module):
    """
    Postprocessing to sample out tokens and prepare the next iteration's inputs. This includes:
    1. Sampling on logits-to-token index, including greedy search.
    2. Update attention_mask's last_token_index position to be 1. Attention mask is used by token_gen only.
    3. Update the generated tokens in-place at the position of last_token_index.
    4. Update the generated index in-place

    On RDU dataflow, the load of last_token_index moves data from DDR to SRAM. This happends only once and then
    the SRAM is consumed by increment and attention_mask. The output of increment is written into a different
    SRAM address and then written further back into the same DDR address of the input by region sharing.
    """
    def __init__(self,
                 sampling_method: SamplingMethod,
                 use_plugin_heuristics: bool = False,
                 run_early_tp: bool = False,
                 hyperfunction: Optional['HyperfunctionForCausalLM'] = None):
        super().__init__()
        self.use_plugin_heuristics = use_plugin_heuristics
        self.run_early_tp = run_early_tp
        self.sampling_method = sampling_method
        self.sampling_module = SamplingModule.get_registered_module(sampling_method)()
        self.token_count_update = None
        self.hyperfunction = hyperfunction

    def update_token_count(self, token_count, next_tokens):
        """Updates the token_count based on newly generated tokens.
        Args:
            token_count: the count of tokens generated in each sample of the batch
            next_tokens: the next tokens that are generated"""
        if self.token_count_update is None:
            self.token_count_update = torch.ones([next_tokens.shape[0], 1, 1], dtype=torch.int32)

        updated_token_count = scatter(
            operand=token_count,
            update=self.token_count_update,  # BS, SS, 1
            start_indices=next_tokens.unsqueeze(-1),  # generated_tokens.shape, 1
            scatter_dims=[0],
            rmw_op="kAdd",
            batched_dims=1)
        return updated_token_count

    def forward_token_gen_length_eq_one(self, tokens, last_token_index, attention_mask, generated_tokens,
                                        generated_tokens_streaming, generated_index, token_count):
        """
        Perform a single step of token generation when the generation length is equal to one.

        This method updates the attention mask, generated tokens, and token count based on the input tokens and their indices.

        Args:
            tokens (Tensor): The input tokens.
            last_token_index (Tensor): The index of the last token in the sequence.
            attention_mask (Tensor): The attention mask for the sequence.
            generated_tokens (Tensor): The generated tokens so far.
            generated_tokens_streaming (Tensor): The generated tokens in streaming mode.
            generated_index (Tensor): The index of the generated tokens.
            token_count (Tensor): The count of tokens generated so far.

        Returns:
            A dictionary containing the updated values:
                - 'tokens': The input tokens (unchanged).
                - 'last_token_index': The updated last token index.
                - 'attention_mask': The updated attention mask.
                - 'generated_tokens': The updated generated tokens.
                - 'generated_tokens_streaming': The updated generated tokens in streaming mode.
                - 'generated_index': The updated generated index.
                - 'token_count': The updated token count.

        Note:
            This method assumes that the input tensors have the correct shape and type.
            The batch size is assumed to be `self.batch_size`, and the maximum sequence length is `self.max_seq_length`.
        """
        last_token_index = last_token_index + 1
        # Cap last_token_index here so that a short prompt can still keep generating even if a longer prompt in the
        # same batch reached max_seq_length. Capping is important to prevent out-of-bound scatter which has undefined
        # behavior on RDU.
        last_token_index = torch.minimum(last_token_index, torch.full((self.batch_size, 1), self.max_seq_length - 1))

        update = torch.ones(self.batch_size, 1, 1).int()
        start_indices = last_token_index.unsqueeze(1)
        attention_mask = scatter(attention_mask, update, start_indices, scatter_dims=[0], batched_dims=1)

        scattering_tokens = tokens.reshape(-1, 1, 1)
        start_indices = generated_index.unsqueeze(1)
        generated_tokens = scatter(generated_tokens, scattering_tokens, start_indices, scatter_dims=[0], batched_dims=1)
        generated_index = generated_index + 1
        generated_index = torch.minimum(generated_index, torch.tensor([self.max_seq_length - 1]))

        updated_token_count = self.update_token_count(token_count, tokens)
        generated_tokens_streaming = scatter(generated_tokens_streaming,
                                             scattering_tokens,
                                             start_indices,
                                             scatter_dims=[0],
                                             batched_dims=1)

        return {
            'tokens': tokens,
            'last_token_index': last_token_index,
            'attention_mask': attention_mask,
            'generated_tokens': generated_tokens,
            'generated_tokens_streaming': generated_tokens_streaming,
            'generated_index': generated_index,
            'token_count': updated_token_count,
        }

    def forward_token_gen_length_gt_one(self, tokens, last_token_index, attention_mask, generated_tokens,
                                        generated_tokens_streaming, generated_index, input_ids):
        """
        Perform a forward pass for token generation when the sequence length is greater than one.

        This method updates the attention mask, generated tokens, and indices based on the input tokens and their corresponding
        attention mask. It uses speculative decoding to generate tokens in parallel and then selects the correct tokens based
        on the generated indices. Repetition penalty is not supported for speculative decoding.

        Args:
            tokens (Tensor): The input tokens.
            last_token_index (Tensor): The index of the last token in the sequence.
            attention_mask (Tensor): The attention mask for the input tokens.
            generated_tokens (Tensor): The generated tokens so far.
            generated_tokens_streaming (Tensor): The generated tokens in streaming mode.
            generated_index (Tensor): The index of the generated tokens.
            input_ids (Tensor): The input IDs.

        Returns:
            A dictionary containing the updated tokens, last token index, attention mask, generated tokens, generated tokens
            in streaming mode, and generated index.

        Returns:
            dict: A dictionary containing the following keys:
                - 'tokens': The input tokens.
                - 'last_token_index': The updated index of the last token in the sequence.
                - 'attention_mask': The updated attention mask.
                - 'generated_tokens': The updated generated tokens.
                - 'generated_tokens_streaming': The updated generated tokens in streaming mode.
                - 'generated_index': The updated index of the generated tokens.
        """
        assert len(attention_mask.shape) == 4 and attention_mask.shape[0] == self.batch_size and attention_mask.shape[
            -2] == self.token_gen_seq_length and attention_mask.shape[
                1] == 1, "attention_mask for token generation has to be of shape (batch_size, 1, k+1, max_seq_length)"

        speculative_decoding_k = self.token_gen_seq_length - 1

        from sambaflow.samba.functional.stir import sn_imm, sn_iteridx, sn_select, sn_zipmapreduce
        from sambaflow.samba.sn_private import sn_gather, sn_scatter
        from sambaflow.samba.utils import SNType

        # gather the generated indices so that they are slightly shifted
        shift_address = from_torch_tensor(torch.minimum(
            torch.arange(1, speculative_decoding_k + 1 + 1, dtype=torch.int),
            torch.tensor([input_ids.shape[-1] - 1], dtype=torch.int)).expand(input_ids.shape).unsqueeze(-1),
                                          name='shift_input_ids_tensor')  # [1, speculative_decoding_k + 1 + 1]

        gathered_index = sn_gather(input_ids, shift_address, [0], [1], batched_dims=1).squeeze(-1)

        # compute the number of accepted tokens
        def update_generated_index(attrs, draft_model_tokens, target_model_tokens):
            nonlocal speculative_decoding_k
            return sn_select(draft_model_tokens == target_model_tokens,
                             sn_imm(speculative_decoding_k + 1, dtype=SNType.INT32),
                             sn_iteridx(attrs=attrs, dim=1, dtype=SNType.INT32) + sn_imm(1, dtype=SNType.INT32))

        selected_index = sn_zipmapreduce(update_generated_index, [gathered_index, tokens],
                                         reduce_func="MIN",
                                         reduce_dim=[1])  # [batch_size, 1]

        def pp_add(attrs, a, b):
            return a + b

        # helper matrix to progress the attention mask
        addition_matrix = torch.arange(1, speculative_decoding_k + 1 + 1, dtype=torch.int).expand(input_ids.shape)
        addition_matrix = addition_matrix[:, None, :].repeat(1, speculative_decoding_k + 1, 1)
        addition_matrix_rdu = from_torch_tensor(
            addition_matrix,
            name='addition_matrix_rdu')  # [batch_size, speculative_decoding_k + 1, speculative_decoding_k + 1]

        # progress attention mask
        generated_index_3d = selected_index.unsqueeze(-1)  # [batch_size, 1, 1]
        attention_mask_progression = torch.minimum(addition_matrix_rdu, generated_index_3d)
        start_indices = last_token_index.unsqueeze(-1) + attention_mask_progression
        update = torch.full(
            (self.batch_size, 1, speculative_decoding_k + 1, speculative_decoding_k + 1, 1),
            0,
            dtype=attention_mask.dtype)  # [batch_size, 1, speculative_decoding_k + 1, speculative_decoding_k + 1]

        start_indices = start_indices.unsqueeze(-1).unsqueeze(1)
        attention_mask = sn_scatter(attention_mask, update, start_indices, scatter_dims=[0], batched_dims=3)

        # progress index
        last_token_index = sn_zipmapreduce(pp_add, [last_token_index, selected_index])
        # Cap last_token_index here so that shorter prompt can still keep generating even if the longer one in the
        # same batch reached max_seq_length. Capping is important to prevent out-of-bound scatter which has undefined
        last_token_index = torch.minimum(last_token_index, torch.full((self.batch_size, 1), self.max_seq_length - 1))

        scatter_matrix = from_torch_tensor(torch.arange(0, speculative_decoding_k + 1,
                                                        dtype=torch.int).repeat(self.batch_size, 1),
                                           name="generated_tokens_scatter_matrix")  # [bs, k + 1]
        scatter_index = sn_zipmapreduce(pp_add, [generated_index, scatter_matrix])  # [bs, k + 1]
        scattering_tokens = tokens.reshape(-1, self.token_gen_seq_length, 1)
        generated_tokens = scatter(generated_tokens,
                                   scattering_tokens,
                                   scatter_index.unsqueeze(-1),
                                   scatter_dims=[0],
                                   batched_dims=1)

        generated_index = sn_zipmapreduce(pp_add, [generated_index, selected_index])
        # Cap last_token_index here so that shorter prompt can still keep generating even if the longer one in the
        # same batch reached max_seq_length. Capping is important to prevent out-of-bound scatter which has undefined
        generated_index = torch.minimum(generated_index, torch.full((self.batch_size, 1), self.max_seq_length - 1))

        generated_tokens_streaming = scatter(generated_tokens_streaming,
                                             scattering_tokens,
                                             scatter_index.unsqueeze(-1),
                                             scatter_dims=[0],
                                             batched_dims=1)

        return {
            'tokens': tokens,
            'last_token_index': last_token_index,
            'attention_mask': attention_mask,
            'generated_tokens': generated_tokens,
            'generated_tokens_streaming': generated_tokens_streaming,
            'generated_index': generated_index,
        }

    def forward(
            self,
            logits: torch.Tensor,
            last_token_index: torch.Tensor,
            attention_mask: torch.Tensor,
            generated_tokens: torch.Tensor,
            generated_tokens_streaming: torch.Tensor,
            generated_index: torch.Tensor,
            temperature: Optional[torch.Tensor] = None,
            top_k: Optional[torch.Tensor] = None,
            top_p: Optional[torch.Tensor] = None,
            pre_generated_randoms: Optional[torch.Tensor] = None,
            repetition_penalty: Optional[torch.Tensor] = None,
            token_count: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Logits tensor of token to be generated, (batch_size, 1, vocab_size).
            attention_mask: Attenion mask to be used by token_gen graph only, (batch_size, max_seq_length).
            last_token_index: The index positions of the last non-pad tokens of input_ids, (batch_size, 1).
            generated_tokens: All generated tokens, to be updated with new tokens at the current position
                              (generated_index + 1), of shape (batch_size, max_seq_length)
            generated_tokens_streaming: The first few generated tokens, of shape (batch_size, kMAX_STREAMING_TOKENS)
            generated_index: The index (starting at 0) to put the generated_tokens, (batch_size, 1).
            temperature: Temperature for multinomial sampling, of shape (batch_size, 1).
            top_k: Top k for top-k sampling, of shape (batch_size, 1).
            top_p: Top p for top-p sampling, of shape (batch_size, 1).
            pre_generated_randoms: Pre-generated randoms for sn_multinomial sampling, of shape (batch_size,
                max_seq_length).
            repetition_penalty: repetition penalty, of shape (batch_size, 1).
            token_count: count of previously generated tokens, of shape (batch_size, vocab_size).
            input_ids: spec decoding token gen input of draft tokens, of shape (batch_size, k + 1).

        Returns:
            Sampled tokens and other in-place updated inputs.
        """
        # hyperfunction._fuse_lm_head_with_sampling is toggled with the `fuse_lm_head_with_sampling` context manager
        if self.hyperfunction is not None and self.hyperfunction._fuse_lm_head_with_sampling:
            hyperfunction = self.hyperfunction.fused_cls_postprocess_hyperfunction
        else:
            if self.run_early_tp == True:
                heuristic = "kPostprocess_SPMD"
                plugins = ["libSPMD_PostprocessPlugins.so"]
            else:
                heuristic = {"distribution": "kPostprocess", "tiling": "kPostprocess", "mapping": "kPostprocess"}
                plugins = ["libPostprocessPlugins.so"]
            hyperfunction = op_fusion(func_name=f"postprocess_{self.sampling_method.name}",
                                      heuristic=heuristic,
                                      plugins=plugins,
                                      user_annotated=self.use_plugin_heuristics)

        with hyperfunction:
            self.batch_size = logits.shape[0]
            self.max_seq_length = generated_tokens.shape[-1]
            assert len(logits.shape) == 3, "logits has to be sliced of shape (batch_size, k + 1, vocab_size)"
            self.token_gen_seq_length = logits.shape[1]
            assert list(last_token_index.shape) == [
                self.batch_size, self.token_gen_seq_length
            ], "last_token_index has to be of shape (batch_size, token_gen_seq_length)"
            assert list(generated_index.shape) == [self.batch_size,
                                                   1], "generated_index has to be of shape (batch_size, 1)"
            assert repetition_penalty is None or list(repetition_penalty.shape) == [
                self.batch_size, 1
            ], "repetition_penalty has to be of shape (batch_size, 1)"

            if self.sampling_method == SamplingMethod.greedy:
                tokens = self.sampling_module(logits, repetition_penalty, token_count)
            elif self.sampling_method == SamplingMethod.multinomial:
                tokens = self.sampling_module(logits, last_token_index, temperature, top_k, top_p,
                                              pre_generated_randoms, repetition_penalty, token_count)
            else:
                raise ValueError(f"Unsupported sampling method: {self.sampling_method.name}")

            if self.token_gen_seq_length == 1:
                return self.forward_token_gen_length_eq_one(tokens, last_token_index, attention_mask, generated_tokens,
                                                            generated_tokens_streaming, generated_index, token_count)
            else:
                return self.forward_token_gen_length_gt_one(tokens, last_token_index, attention_mask, generated_tokens,
                                                            generated_tokens_streaming, generated_index, input_ids)


class CausalModelWithSampling(torch.nn.Module):
    """
    A wrapper model that includes ModelForCausalLM and PostprocessWithSampling. This wrapper model expects the
    lm_model to return hidden states after the last encoder layer (before any normalization) so the remaining graph can
    be fused with the postprocessing graph. The remaining graph usually includes norm and lm_head.
    # TODO: what are norm and lm_head?
    """
    def __init__(self, lm_model: torch.nn.Module, sampling_method: SamplingMethod):
        super().__init__()
        self.lm_model = lm_model
        self.postprocess = PostprocessWithSampling(sampling_method,
                                                   use_plugin_heuristics=lm_model.config.use_plugin_heuristics,
                                                   hyperfunction=self.lm_model.hyperfunction)

    def forward(
            self,
            token_gen_inputs: Dict[str, Any],
            last_token_index: torch.Tensor,
            attention_mask: torch.Tensor,
            generated_tokens: torch.Tensor,
            generated_tokens_streaming: torch.Tensor,
            generated_index: torch.Tensor,
            temperature: Optional[torch.Tensor] = None,
            top_k: Optional[torch.Tensor] = None,
            top_p: Optional[torch.Tensor] = None,
            pre_generated_randoms: Optional[torch.Tensor] = None,
            repetition_penalty: Optional[torch.Tensor] = None,
            token_count: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        last_token_index and attention_mask are duplicates of those in token_gen_inputs because we only want to
        trace the postprocessing graph part that's connected to last_token_index and attention_mask, not any nodes
        from token_gen.

        Args:
            token_gen_inputs: Inputs for token generation graph of cached inference.
            attention_mask: Attenion mask to be updated, (batch_size, max_seq_length).
            last_token_index: The index positions of the last non-pad tokens of input_ids, (batch_size, 1).
            generated_tokens: All generated tokens, to be updated with new token at the current position
                              (generated_index + 1), of shape (batch_size, max_seq_length)
            generated_tokens_streaming: The first few generated tokens, of shape (batch_size, kMAX_STREAMING_TOKENS)
            generated_index: The index starting from 0 to put the generated_tokens, (batch_size, 1)..
            temperature: Temperature for multinomial sampling, of shape (batch_size, 1).
            top_k: Top k for top-k sampling, of shape (batch_size, 1).
            top_p: Top p for top-p sampling, of shape (batch_size, 1).
            pre_generated_randoms: Pre-generated randoms for sn_multinomial sampling, of shape (batch_size,
                max_seq_length).
            repetition_penalty: repetition penalty, of shape (batch_size, 1).
            token_count: count of previously generated tokens, of shape (batch_size, vocab_size).

        Returns:
            Sampled tokens, in-place updated inputs and last_hidden_states for slicing the fused postprocess graph.
        """
        logits, *_, last_hidden_states = self.lm_model(**token_gen_inputs)
        outputs = self.postprocess(logits, last_token_index, attention_mask, generated_tokens,
                                   generated_tokens_streaming, generated_index, temperature, top_k, top_p,
                                   pre_generated_randoms, repetition_penalty, token_count)
        # last_hidden_states must be appended at the end so tracing outputs can retrieve it from tuple
        outputs['last_hidden_states'] = last_hidden_states
        return outputs


class ModelWithPostprocessInit(torch.nn.Module):
    """
    A wrapper model that includes ModelForCausalLM and PostprocessInit. This wrapper model runs the cache_gen graph and
    the PostprocessInit module to initialize the token_count tensor with the input_ids.
    Args:
        lm_model: the language model
        batch_size: the batch size
        vocab_size: the vocab size
    """
    def __init__(self, lm_model: torch.nn.Module, batch_size: int, vocab_size: int, seq_length: int):
        super().__init__()
        self.lm_model = lm_model
        self.postprocess_init = PostprocessInit(batch_size, vocab_size)
        self.seq_length = seq_length

    def forward(
            self,
            cache_gen_inputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor, torch.Tensor]:
        """
        Runs the cache_gen graph and the PostprocessInit module to initialize the token_count tensor with the input_ids.

        Args:
            cache_gen_inputs: the dictionary of inputs to the cache_gen graph

        Returns:
            The cache_gen graph's outputs (logits, KV cache, and last hidden states) and token_count
        """
        outputs = self.lm_model(**cache_gen_inputs)
        token_count = None
        if cache_gen_inputs["last_token_index"].shape[-1] == 1:
            context = self.lm_model.hyperfunction.classifier(
                self.seq_length, False, False,
                reuse_last_id=True) if hasattr(self.lm_model, "hyperfunction") and hasattr(
                    self.lm_model.hyperfunction, "classifier") else nullcontext()
            with context, opfusion_id('postprocess_init'):
                token_count = self.postprocess_init(cache_gen_inputs['input_ids'], cache_gen_inputs['last_token_index'])
        return (*outputs, token_count)
