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

import hydra
import torch
from accelerate.utils import set_seed
from config.schema import CPUGenerationAppConfig
from sambanova_modelzoo.libs.common.arguments import to_pydantic
from sambanova_modelzoo.libs.nlp.core.generation.configuration_utils import (configure_pad_token,
                                                                             get_config_overrides_for_generation)
from sambanova_modelzoo.models.utils import load_model_from_pretrained
from transformers import AutoTokenizer


@hydra.main(config_path="config", config_name="base_config_cpu")
@to_pydantic(CPUGenerationAppConfig)
def main(cfg: CPUGenerationAppConfig):
    # For reproducibility of pytorch
    set_seed(cfg.generation.seed)
    torch.use_deterministic_algorithms(True)
    # SN model only supports right padding
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint.model_name_or_path, padding_side='right')
    # Use EOS token for models that do not have padding tokens in the checkpoint
    configure_pad_token(tokenizer)
    print(f"Prompts:\n{cfg.generation.prompts}")
    inputs = tokenizer(cfg.generation.prompts, padding=True, return_tensors='pt')
    print(f"Tokenized inputs:\n{inputs}")
    print('Loading ckpt ...')
    original_config_overrides = get_config_overrides_for_generation(pad_token_id=tokenizer.pad_token_id)
    sn_model = load_model_from_pretrained(cfg.checkpoint.model_name_or_path, cfg.model, original_config_overrides)
    sn_model.eval()

    input_length = len(inputs['input_ids'][0])
    max_new_tokens = cfg.generation.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_seq_length - input_length
    sn_model.init_inputs_processor(static_seq_lengths=set(cfg.generation.static_seq_lengths))
    print(f'Generating {max_new_tokens} tokens ...')
    outputs = sn_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_tokens = outputs[:, input_length:]
    print('Decoding ...')
    completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Completion:\n{completions}")


if __name__ == '__main__':
    main()
