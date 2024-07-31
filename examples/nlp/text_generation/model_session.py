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

import time
from contextlib import contextmanager, nullcontext
from typing import Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import sambaflow.samba as samba
from sambaflow.samba import SambaTensor
from sambaflow.samba.profiler import Profiler


@contextmanager
def profiler(msg: str):
    st = time.time()
    yield
    print(f"{msg} {time.time() - st} seconds")


@contextmanager
def generate_forward(model: torch.nn.Module, logits_name: str, samba_profiler: Profiler, profile: bool = False):
    """
    A decorator that swaps the model forward call with samba.session.run using the compiled PEF and return the outputs
    in a format to work with Hugging Face model.generate() function. The forward call of the model is achieved through
    `model()`.
    Args:
        model: SNModel to be called for text generation.
        logits_name: Traced logit output region name that can be used to retrieve data from RDU device memory.
        profile: Whether to profile the generation.
    """
    def rdu_forward(model: torch.nn.Module, *args, **kwargs) -> Tuple[SambaTensor]:
        profile_fwd = samba_profiler.start_event('fwd')
        samba_inputs = samba.from_named_dict({name: t for name, t in kwargs.items() if isinstance(t, torch.Tensor)})
        cache_gen = model.inputs_processor.preprocess_inputs_for_token_gen.calls == 0
        with profiler(f"cache_gen={cache_gen} session.run") if profile else nullcontext():
            samba.session.run(input_tensors=samba_inputs,
                              graph_name=model.inputs_processor.graph_to_call(cache_gen, model.chosen_length))
            logits = samba.session.get_tensors_by_name([logits_name])[0].data
        samba_profiler.end_event(profile_fwd)
        return CausalLMOutputWithCrossAttentions(loss=None, logits=logits)

    torch_forward = model.__class__.__call__
    model.__class__.__call__ = rdu_forward
    yield
    model.__class__.__call__ = torch_forward
