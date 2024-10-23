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
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from config.schema import RDUGenerationAppConfig
from model_session import generate_forward
from sambanova_modelzoo.libs.common.arguments import to_pydantic
from sambanova_modelzoo.libs.nlp.core.generation.cached_inference_compiler import CachedInferenceCompiler
from sambanova_modelzoo.libs.nlp.core.generation.configuration_utils import (configure_pad_token,
                                                                             get_config_overrides_for_generation)
from sambanova_modelzoo.models.configuration_validator import SNAutoConfigValidator
from sambanova_modelzoo.models.utils import load_model_from_config, load_model_from_pretrained
from utils.reporting import save_summary_report
from transformers import AutoTokenizer

import sambaflow.samba as samba
from sambaflow.samba.profiler import Profiler


def compile(cfg: RDUGenerationAppConfig) -> str:
    """
    Compile the model
    Args:
        cfg: Parsed Pydantic model from yaml file
    Returns:
        Compiled pef file name
    """
    original_config_overrides = get_config_overrides_for_generation()
    with init_empty_weights():
        sn_model = load_model_from_config(cfg.checkpoint.model_name_or_path, cfg.model, original_config_overrides)
    # Will error out if the input configuration is not supported by SambaNova. Add `job_config.validate_config=False` to bypass this check
    SNAutoConfigValidator.validate(model_config=sn_model.config, job_config=cfg)
    sn_model.eval()
    compiler = CachedInferenceCompiler(sn_model, cfg.generation.batch_size, set(cfg.generation.static_seq_lengths))
    return compiler.compile(cfg=cfg.samba_compile.model_dump())


def run(cfg: RDUGenerationAppConfig):
    """
    Run the compiled model (pef)
    Args:
        cfg: parsed Pydantic model from yaml file
    """
    # enable SambaProfile for ttft and friends
    samba_profile = Profiler()
    samba_profile.start_profile()

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
    # Trace again to get the output SambaTensor container and setup the multigraph
    compiler = CachedInferenceCompiler(sn_model, cfg.generation.batch_size, set(cfg.generation.static_seq_lengths))
    compiler.trace()

    # Initialize the RDU memory with model weights
    samba.session.init_multigraph_runtime(cfg.samba_run.pef)

    input_length = len(inputs['input_ids'][0])
    max_new_tokens = cfg.generation.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_seq_length - input_length
    print(f'Generating {max_new_tokens} tokens ...')
    # Samba runner prepares input names to match compile time input names
    sn_model.init_inputs_processor(runner='samba', static_seq_lengths=set(cfg.generation.static_seq_lengths))
    with generate_forward(sn_model,
                          logits_name=compiler.input_names.logits,
                          samba_profiler=samba_profile,
                          profile=cfg.generation.profile):
        outputs = sn_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_tokens = outputs[:, input_length:]
    print('Decoding ...')
    completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    ttft = samba_profile._get_iter_times('fwd', 0)[0]
    tokens_ex = samba_profile.get_latencies(['fwd'], 1).get('fwd')

    summary_text = (
        f"Completion:\n{completions}\n"
        f"latencies\n"
        f"    time to first token {ttft:.4f}s\n"
        f"    tokens,  excluding first token {tokens_ex:.4f}s\n"
        f"    tokens,  overall {samba_profile.get_latencies(['fwd'], 0).get('fwd'):.4f}s\n"
        f"    Total Latency {(ttft + (tokens_ex * len(generated_tokens))):.4f}s\n"
        f"throughputs\n"
        f"    tokens/second excluding first token {samba_profile.get_throughputs(['fwd'], cfg.generation.batch_size, 1).get('fwd'):.4f}\n"
        f"    tokens/second overall {samba_profile.get_throughputs(['fwd'], cfg.generation.batch_size, 0).get('fwd'):.4f}\n"
    )
    print(summary_text)
    save_summary_report(cfg.generation.output_dir, 'summary.txt', summary_text)
    print(f'Summary saved at {cfg.generation.output_dir}/summary.txt')



    samba_profile.stop_profile()


@hydra.main(config_path="config", config_name="base_config_rdu")
@to_pydantic(RDUGenerationAppConfig)
def main(cfg: RDUGenerationAppConfig):
    set_seed(cfg.generation.seed)
    if cfg.command == 'compile':
        samba.session.setup(cfg.samba_compile)
        compile(cfg)
    elif cfg.command == 'run':
        samba.session.setup(cfg.samba_run)
        run(cfg)


if __name__ == '__main__':
    main()
