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

from sambanova_modelzoo.libs.nlp.core.clm_tracer import PretrainTracer
from sambanova_modelzoo.models.llama.trace_llama import LlamaTracer

from .modeling_gemma import SNGemmaForCausalLM

"""Gemma inputs are the same as Llama inputs"""


class GemmaTracer(LlamaTracer, model=SNGemmaForCausalLM):
    pass


class GemmaPretrainTracer(PretrainTracer, model=SNGemmaForCausalLM):
    pass
