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

from enum import Enum
from typing import Dict, Final, Type

import torch
import torch.nn.functional as F

kMAX_K_VALUE: Final[int] = 128


class SamplingMethod(Enum):
    """
    Sampling methods supported inside the PEF, can be used to select the graph call
    # TODO: Which graph call? What are options here?
    """
    greedy = 'greedy'
    multinomial = 'sample'


class SamplingModule(torch.nn.Module):
    method_to_module: Dict[SamplingMethod, Type['SamplingModule']] = {}

    def __init_subclass__(cls, method: SamplingMethod = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if method is None:
            raise RuntimeError('a method to register SamplingModule is required')
        SamplingModule.method_to_module[method] = cls

    @classmethod
    def get_registered_module(cls, method: SamplingMethod) -> 'SamplingModule':
        """
        Args:
            method: Name of the sampling method to retrieve.
        Returns:
            The registered module class of that corresponds to the specified sampling method.
        """
        if method not in cls.method_to_module:
            raise ValueError(
                f"{method} is not registered with SamplingModule. Registered are {cls.method_to_module.keys()}")
        return cls.method_to_module[method]

    @classmethod
    def is_fused_with_lm_head(cls) -> bool:
        """ Whether the sampling module should be fused with LM head """
        return True

    @classmethod
    def is_ready_for_deployment(cls) -> bool:
        """ Whether the sampling module should be exposed to models """
        return True


class GreedySampling(SamplingModule, method=SamplingMethod.greedy):
    """
    Given sliced logits, output the most likely token (logit index). Also supports repetition penalty.
    """
    def __init__(self):
        super().__init__()
        self.repetition_penalty_processor = SNRepetitionPenaltyLogitsProcessor()

    def forward(self, logits: torch.Tensor, repetition_penalty: torch.Tensor,
                token_count: torch.Tensor) -> torch.Tensor:
        assert len(
            logits.shape) == 3 and logits.shape[1] == 1, "logits must be sliced to be of shape (bs, 1, vocab_size)"

        logits = self.repetition_penalty_processor(logits.squeeze(1), repetition_penalty, token_count)
        logits = logits.unsqueeze(1)
        next_token = torch.argmax(logits, dim=-1).int()
        return next_token


class SNRepetitionPenaltyLogitsProcessor(torch.nn.Module):
    """
    Logits processor that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.
    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.
    This processor maintains a ``repetition_penalty_mask`` that keeps track of which tokens the model has encountered in the
    input prompt or has generated. After a token is generated, you should call :meth:`update_mask` with the generated
    token to update the ``repetition_penalty_mask``.
    """
    def __init__(self):
        super().__init__()

        self.repetition_penalty_mask = None  # (BS, vocab_size) that is a count of which tokens have appeared in the prompt or the generation text
        self.mask_value = None

    def forward(self, logits: torch.Tensor, penalty: torch.Tensor, token_count: torch.Tensor):
        """Applies the repetition penalty.
        Args:
            logits: the original logits, of shape (batch_size, vocab_size)
            penalty: the penalty to apply, of shape (batch_size, 1). If ``None``, this function is a no-op.
            token_count: the count of each encountered token, including prompt tokens and generated tokens, of shape
                (batch_size, vocab_size).
        Returns:
            the logits with repetition penalty applied
        """
        from sambanova_modelzoo.models.custom_ops import sn_imm, sn_select, sn_zipmapreduce

        # Update the logits in streaming fashion
        r_penalty = torch.reciprocal(penalty).float()
        logits = logits.float()
        penalty = penalty.float()

        logits = sn_zipmapreduce(
            lambda attrs, logits, token_count, penalty, r_penalty: sn_select(
                token_count > sn_imm(0, dtype=torch.int32),
                sn_select(logits > sn_imm(0, dtype=torch.float32), logits * r_penalty, logits * penalty), logits),
            [logits, token_count, penalty, r_penalty])

        return logits


class MultinomialSampling(SamplingModule, method=SamplingMethod.multinomial):
    """
    Implements a multinomial sampling module for sequence generation tasks. The module
    supports temperature scaling, top-k and top-p probability filtering, and repetition penalty.
    This module scales logits based on a given temperature and uses top-k and top-p filtering to
    limit sampling to the most probable outcomes. This module also uses repetition penalty to prevent the same tokens
    from being generated too many times.
    Init Args:
        max_k: the maximum top_k value allowed.
        use_repetition_penalty: whether to use repetition penalty or not. If you know ahead of time that you will not
            use repetition penalty, specify ``False`` to skip the repetition penalty logits processor.

    Args:
        logits: Logits tensor of token to be generated, (batch_size, 1, vocab_size).
        generated_index: The index (starting at 0) to put the generated_tokens, (batch_size, 1).
        temperature: Temperature for multinomial sampling, of shape (batch_size, 1).
        top_k: Top k for top-k sampling, of shape (batch_size, 1).
        top_p: Top p for top-p sampling, of shape (batch_size, 1).
        pre_generated_randoms: Pre-generated randoms for sn_multinomial sampling, of shape (batch_size, max_seq_length).
        repetition_penalty: penalty for repeated tokens, of shape (batch_size, 1). If ``None``, skips the repetition
            penalty processor, but still updates the repetition penalty mask with the newly generated token.
        token_count: count of previously generated tokens, of shape (batch_size, vocab_size).

    Returns:
        torch.Tensor: The indices of the next tokens sampled, shaped (batch_size, 1).
    """
    def __init__(self, max_k: int = kMAX_K_VALUE):
        super(MultinomialSampling, self).__init__()
        self.max_k = max_k
        self.repetition_penalty_processor = SNRepetitionPenaltyLogitsProcessor()

    def forward(self, logits: torch.Tensor, generated_index: torch.Tensor, temperature: torch.Tensor,
                top_k: torch.Tensor, top_p: torch.Tensor, pre_generated_randoms: torch.Tensor,
                repetition_penalty: torch.Tensor, token_count: torch.Tensor) -> torch.Tensor:
        # assert inputs shapes
        assert len(logits.shape) == 3, f"Expected logits to be of shape (batch_size, 1, vocab_size), got {logits.shape}"
        assert len(generated_index.shape
                   ) == 2, f"Expected generated_index to be of shape (batch_size, 1), got {generated_index.shape}"
        assert len(
            temperature.shape) == 2, f"Expected temperature to be of shape (batch_size, 1), got {temperature.shape}"
        assert len(top_k.shape) == 2, f"Expected top_k to be of shape (batch_size, 1), got {top_k.shape}"
        assert len(top_p.shape) == 2, f"Expected top_p to be of shape (batch_size, 1), got {top_p.shape}"
        assert len(
            pre_generated_randoms.shape
        ) == 2, f"Expected pre_generated_randoms to be of shape (batch_size, sequence_length), got {pre_generated_randoms.shape}"

        batch_size, _, vocab_size = logits.shape
        # max_k should be divisible by 16 and vocab_size should be divisible by max_k
        assert self.max_k % 16 == 0, f"max_k should be a multiple of 16, got {self.max_k}"
        assert vocab_size % self.max_k == 0, f"vocab_size should be divisible by max_k, got {vocab_size} and {self.max_k}"
        # assert batch dimension are equal
        assert batch_size == temperature.shape[0] == top_k.shape[0] == top_p.shape[0] == pre_generated_randoms.shape[
            0], f"Batch sizes do not match, got {batch_size}, {temperature.shape[0]}, {top_k.shape[0]}, {top_p.shape[0]}, {pre_generated_randoms.shape[0]}"

        from sambanova_modelzoo.models.custom_ops import gather as sn_gather
        from sambanova_modelzoo.models.custom_ops import sn_imm, sn_iteridx, sn_multinomial, sn_select, sn_zipmapreduce

        # logits -> (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
        logits = logits.squeeze(1)
        logits = self.repetition_penalty_processor(logits, repetition_penalty, token_count)

        # Temperature scaling
        recip_temperature = torch.reciprocal(temperature)
        logits = logits * recip_temperature

        # TopK fixed to self.max_k to avoid dynamic output shape
        top_logits, top_indices = torch.topk(logits, self.max_k, dim=1, largest=True, sorted=True)
        top_indices = top_indices.to(torch.int32)

        # logits -> (bs, vocab_size)
        top_logits = top_logits.float()
        top_k_logits = sn_zipmapreduce(
            lambda attrs, logits, top_k: sn_select(
                sn_iteridx(dim=1, attrs=attrs, dtype=torch.int32) < sn_select(top_k < sn_imm(
                    1, dtype=torch.int32), sn_imm(1, dtype=torch.int32), top_k), logits,
                sn_imm(torch.finfo(torch.float).min, dtype=torch.float32)), [top_logits, top_k])

        probs = F.softmax(top_k_logits, dim=1)

        # Draw samples using sn_multinomial
        samples = sn_gather(pre_generated_randoms, generated_index, [0], [1], batched_dims=1)
        samples = samples.float()
        top_p = top_p.float()

        scaled_samples = sn_zipmapreduce(
            lambda attrs, sample, top_p: sample * sn_select(
                top_p < sn_imm(0.0, dtype=torch.float32), sn_imm(0.0, dtype=torch.float32),
                sn_select(top_p > sn_imm(1.0, dtype=torch.float32), sn_imm(1.0, dtype=torch.float32), top_p)),
            [samples, top_p])

        # --> [bs]
        sampled_indices = sn_multinomial(probs.float(), scaled_samples)

        # get the original tokens
        gather_indices = sampled_indices.unsqueeze(1).unsqueeze(1).to(top_indices.dtype)
        next_token = sn_gather(top_indices, gather_indices, [0], [1], batched_dims=1).squeeze(1)

        return next_token

    @classmethod
    def is_ready_for_deployment(cls) -> bool:
        """
        Multinomial Sampling is work in progress
        ##SN: see go/j/PTI-166
        """
        return False
