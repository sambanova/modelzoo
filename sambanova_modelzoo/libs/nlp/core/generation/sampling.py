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
from typing import Dict, Type

import torch
import torch.nn.functional as F


class SamplingMethod(Enum):
    """
    Sampling methods supported inside the PEF, can be used to select the graph call
    """
    greedy = 'greedy'


class SamplingModule(torch.nn.Module):
    method_to_module: Dict[SamplingMethod, Type['SamplingModule']] = {}

    def __init_subclass__(cls, method: SamplingMethod = None, **kwargs):
        """ Register the subclasses """
        super().__init_subclass__(**kwargs)
        if method is None:
            raise RuntimeError('method needs to be specified to register SamplingModule')
        SamplingModule.method_to_module[method] = cls

    @classmethod
    def get_registered_module(cls, method: SamplingMethod) -> 'SamplingModule':
        """
        Args:
            method: The method name of sampling to be queried for.
        Returns:
            The registered module class of the correponding sampling method.
        """
        if method not in cls.method_to_module:
            raise ValueError(
                f"{method} is not registered with SamplingModule. Registered are {cls.method_to_module.keys()}")
        return cls.method_to_module[method]

    @classmethod
    def is_fused_with_lm_head(cls) -> bool:
        """ Whehter the sampling module should be fused with LM head """
        return True


class GreedySampling(SamplingModule, method=SamplingMethod.greedy):
    """
    Given sliced logits, output the most likely token (logit index)
    """
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        assert len(
            logits.shape) == 3 and logits.shape[1] == 1, "logits must be sliced to be of shape (bs, 1, vocab_size)"
        return torch.argmax(logits, dim=-1)


class MultinomialSampling(torch.nn.Module):
    """
    Implements a multinomial sampling module for sequence generation tasks that
    supports temperature scaling, repetition penalty, top-p probability filtering,
    and maintaining a minimum number of tokens.

    This module scales logits based on a given temperature, applies a penalty
    to discourage the repetition of specific tokens, and uses top-p filtering to
    limit sampling to the most probable outcomes. It ensures that at least the
    minimum number of tokens are always considered in the sampling process.

    Attributes:
        temperature: scaling factor for logits, default is 1.0.
        penalty: penalty factor for repeating tokens, default is 1.0.
        top_p: cumulative probability threshold for filtering top tokens, default is 0.1.
        min_tokens_to_keep: minimum number of tokens to consider, default is 1.
        use_encoder_repetition_penalty: defaults to `False` to use repetition_penalty mode.
        In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors
        suggest the use of a penalty of around 1.2 to achieve a good balance between truthful generation
        and lack of repetition. To penalize and reduce repetition, use ``penalty`` values above 1.0,
        where a higher value penalizes more strongly. To reward and encourage repetition, use ``penalty``
        values between 0.0 and 1.0, where a lower value rewards more strongly. 
        If set to `True`, the alternative of repetition_penalty, with the penalty value is inversed.
        In other words, a penalty above 1.0 increases the odds of selecting tokens that were present in
        the prompt. It was designed to avoid hallucination in input-grounded tasks, like summarization.

    Methods:
        forward(logits, input_ids):
            Processes the logits and performs the sampling, returning the indices of the sampled tokens.

    Args:
        logits: the logits to sample from, shaped (batch_size, vocab_size).
        input_ids: indices of previously chosen tokens to penalize, shaped (batch_size, 1).

    Returns:
        torch.Tensor: The indices of the next tokens sampled, shaped (batch_size, 1).

    Raises:
        ValueError: If `logits` does not have the correct shape (batch_size, vocab_size).
    """
    def __init__(self,
                 temperature: float = 1.0,
                 penalty: float = 1.0,
                 top_p: float = 0.1,
                 top_k: int = 1000,
                 min_tokens_to_keep: int = 1,
                 use_encoder_repetition_penalty: bool = False):
        super(MultinomialSampling, self).__init__()
        self.temperature = temperature
        self.top_p = top_p

        MAX_K_VALUE = 1000
        assert top_k <= MAX_K_VALUE, f"SambaNova only supports values of top_k that are less than or equal to {MAX_K_VALUE}, received {top_k}."
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        if use_encoder_repetition_penalty:
            penalty = 1 / penalty
        self.penalty = penalty

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if len(logits.shape) != 2:
            raise ValueError("Logits must be of shape (batch_size, vocab_size)")

        if logits.shape[0] != input_ids.shape[0]:
            raise ValueError("Batch sizes of logits and input_ids do not match.")

        if input_ids.shape[1] != 1:
            raise ValueError("Expected input_ids to have shape (batch_size, 1)")

        _, vocab_size = logits.shape

        # Scale logits by temperature
        logits = logits / self.temperature

        # Apply repetition penalty (input_ids will be padded with padding tokens)
        # Make sure input_ids is broadcastable to the shape of logits
        if input_ids.shape[1] != logits.shape[1]:
            # Assuming input_ids are in the shape (batch_size, 1)
            input_logits = torch.gather(logits, 1, input_ids)
            penalized_logits = torch.where(input_logits < 0, input_logits * self.penalty, input_logits / self.penalty)
            logits = logits.scatter(1, input_ids, penalized_logits)

        # Get top k logits directly in descending order
        k = min(self.top_k, vocab_size)
        top_logits, top_indices = torch.topk(logits, k, dim=1, largest=True, sorted=True)

        # Compute cumulative probabilities
        probs = F.softmax(top_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        mask = cumulative_probs < self.top_p
        mask[:, :self.min_tokens_to_keep] = True  # Ensure minimum tokens are kept
        probs[~mask] = 0
        sampled_indices = torch.multinomial(probs, num_samples=1)
        next_token = torch.gather(top_indices, 1, sampled_indices)
        return next_token
