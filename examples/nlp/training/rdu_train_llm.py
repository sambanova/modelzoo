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

"""
This file demonstrates how to do the following in a RDU environment:
  1. Load an LLM
        - from a checkpoint
        - from a config
  2. Load a dataset
        - that's been processed using generative_data_prep
  3. Train an LLM
        - without using HF trainer
  4. Save a checkpoint

It is for demonstrative purposes only and not intended to be used for real training.
It follows the same structure as and shares much of its code with cpu_train_llm.py
"""

from contextlib import nullcontext
from typing import Dict

import hydra
import torch
from accelerate import init_empty_weights
from config.schema import CheckpointConfig, PretrainedModelConfig, TrainingConfig
from config.schema import  RDUTrainingConfig
from sambanova_modelzoo.libs.common.arguments import to_pydantic
from sambanova_modelzoo.libs.common.pef_meta import APP_ARGS_KEY, from_pef_meta_dict, to_pef_meta_dict
from sambanova_modelzoo.libs.nlp.core.clm_runtime import PretrainRuntime, SambaPretrainInputNames
from sambanova_modelzoo.libs.nlp.core.clm_tracer import PretrainTracer
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformer
from sambanova_modelzoo.models.configuration_validator import SNAutoConfigValidator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, set_seed
from utils.checkpoint import save_as_huggingface_checkpoint
from utils.dataset import HDF5ParallelLoader, get_dataset_metadata
from utils.reporting import save_per_step_report, save_summary_report

from sambaflow import samba
from sambaflow.samba import SambaTensor
from sambaflow.samba.utils import trace_graph
from sambaflow.samba.utils import set_seed as set_samba_seed
from sambaflow.samba.utils.pef_utils import get_pefmeta_dict


def load_model(checkpoint: CheckpointConfig, model_arch: PretrainedModelConfig) -> PreTrainedModel:
    """ Load a Huggingface model, optimized for RDU through ModelZoo """

    # Load the model's Huggingface config.
    # This can be from a config.json file, checkpoint folder or model identifier.
    model_pointer = checkpoint.config_name or checkpoint.model_name_or_path
    config = AutoConfig.from_pretrained(model_pointer)

    # Embed specific model instantiation details using ConfigurationTransformer().
    # Consult config/schema.py for a complete list of parameters.
    modelzoo_params = model_arch.model_dump()
    config = ConfigurationTransformer().run(config, sn_model_args=modelzoo_params)

    # Load model from the config to randomly initialize weights (pretraining) or
    # Load model from checkpoint to load weights from disk (finetuning)
    if checkpoint.config_name:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=model_arch.dtype)
    elif checkpoint.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(checkpoint.model_name_or_path,
                                                     config=config,
                                                     torch_dtype=model_arch.dtype)
    else:
        raise ValueError('checkpoint.config_name or checkpoint.model_name_or_path not provided')

    return model


def load_dataset(training: TrainingConfig, model_seq_length: int) -> DataLoader:
    """ Load a dataset folder prepared by generative_data_prep """

    dataset_path = str(training.dataset)

    # generative_data_prep embeds some metadata in the dataset folder
    # Use it to confirm compatibility with this model
    dataset_seq_length = get_dataset_metadata(dataset_path)['max_seq_length']
    if dataset_seq_length != model_seq_length:
        raise RuntimeError(
            f"This dataset (seq len {dataset_seq_length}) is not compatible with this model (seq len {model_seq_length})"
        )

    # ModelZoo LLMs expect data processed using generative_data_prep.
    # Each sample is {'input_ids': List[List[int]], 'token_type_ids': List[List[int]]}
    # For more details on token_type_ids, refer to TODO: article attention docs
    dataloader = HDF5ParallelLoader(dataset_path, dataset_seq_length, local_batch_size=training.batch_size, drop_last=True)
    return dataloader


def compile(cfg: RDUTrainingConfig, tracing_inputs: Dict[str, "torch.tensor"], model: PreTrainedModel,
            optimizer: samba.optim.AdamW) -> str:
    """ Compile a PEF to run training on RDU """

    # Get the arguments for the compiler
    compile_dict = cfg.samba_compile.model_dump()
    
    # Will error out if the input configuration is not supported by SambaNova. Add `job_config.validate_config=False` to bypass this check
    SNAutoConfigValidator.validate(model_config=model.config, job_config=cfg)
    # Embed some metadata in the PEF.
    # samba_compile metadata is required for `samba.session.setup()` during runtime.
    # But other args (model, checkpoint, training) are purely for convenience
    # (to avoid having the user provide them twice).
    pef_meta = to_pef_meta_dict(cfg)
    print('The following args were embedded into the PEF (along with all Samba args):', pef_meta[APP_ARGS_KEY])

    pef_path = samba.session.compile(
        model,
        tracing_inputs,
        optimizers=optimizer,
        name='rdu_train_llm',
        inference=False,
        config_dict=compile_dict,
        pef_metadata=get_pefmeta_dict(pef_meta, model),
    )

    return pef_path


def create_tracing_inputs(cfg: RDUTrainingConfig, model: AutoModelForCausalLM) -> Dict[str, SambaTensor]:
    """ Create the inputs that are needed to trace a static graph from the torch module"""
    tracer_class = PretrainTracer.get_registered_tracer(type(model))
    tracer = tracer_class(model.config, cfg.training.batch_size)
    return tracer.get_tracing_inputs()


def _move_tensors_to_rdu(model_inputs: Dict[str, torch.tensor], samba_tensor_names: SambaPretrainInputNames):
    """ Convert tensors in a dict to samba tensors with correct model input names """
    for k, v in model_inputs.items():
        tensor_name = samba_tensor_names[k]
        if isinstance(v, tuple):
            model_inputs[k] = tuple(samba.from_torch(t, name=tensor_name[n]) for n, t in enumerate(v))
        else:
            model_inputs[k] = samba.from_torch(v, name=k)


# Read args from base_config_rdu.yaml using Hydra and convert them to RDUTrainingConfig pydantic model
@hydra.main(config_path="config", config_name="base_config_rdu", version_base="1.2")
@to_pydantic(RDUTrainingConfig)
def main(cfg: RDUTrainingConfig):
    # Read compile time information from the PEF metadata
    cfg = from_pef_meta_dict(cfg) if cfg.command == 'run' else cfg

    print('Running app with arguments:')
    print(cfg.model_dump())

    print('Loading model...')
    # Don't load weights in RAM if compiling
    loading_context = init_empty_weights if cfg.command == 'compile' else nullcontext
    with loading_context():
        model = load_model(cfg.checkpoint, cfg.model)
    samba.from_torch_model_(model)  # important! Move model to samba runtime

    print('Loading optimizer...')
    import sambaflow.samba.optim as optim  # important! Load the optimizer from samba instead
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    # NOTE: In the base model source from huggingface, the model computes the loss
    # So there's no need to define a loss function here

    # NOTE: About PEFs and tracing
    # "Tracing" is feeding some inputs to the model, then
    # following/tracing each operation in the forward, backward and optimizer passes.
    # This is built into an execution graph; specific to
    # the particular model, data input, and optimizations of operations.
    # -
    # A PEF is file on disk containing an optimized version of this execution graph.
    # The optimizations are searched for by the compiler during the compile stage.
    # The PEF is read from file during run to
    # tell the runtime how to place each operation on-chip on RDU.

    # Construct dummy input tensors to trace the execution graph
    tracing_inputs = create_tracing_inputs(cfg, model)

    if cfg.command == 'compile':
        # Prepare samba for compile (read compiler info, set env)
        # NOTE: Samba arguments must be consistent for `samba.session.setup()` during compile & run.
        # The arguments can either be embedded in the pef during compile, then read during run;
        # or provided twice but exactly the same during the two stages.
        # This script embeds args in compile().
        samba.session.setup(cfg.samba_compile)

        print('Compiling a PEF...')
        pef_path = compile(cfg, tracing_inputs, model, optimizer)

        print('PEF compiled successfully:', pef_path)
        return
    elif cfg.command == 'run':
        # Prepare samba to run (read runtime version, pef arch info)
        samba.session.setup(cfg.samba_run)

        # Load the optimized execution graph from the PEF (retrace the graph).
        # Model weights are already loaded and moved with `samba.from_torch_model_`.
        # trace_graph returns the output from fwd + backwd, optim
        print('Tracing...')
        model_outputs = trace_graph(model, tracing_inputs, optimizer, pef=cfg.samba_run.pef, loss_indices=[0])
    else:
        raise ValueError('Only compile and run commands are supported')

    # ModelZoo's PretrainRuntime will be used in the training loop to
    # create attention masks and labels from the input data
    inputs_processor = PretrainRuntime(cfg.model.max_seq_length)

    # When tensors are moved to RDU in the traning loop, they need to be assigned a name.
    # Each tensor has a name in the execution graph decided during compile time.
    # The input tensors need to match their unique names at runtime.
    samba_tensor_names = SambaPretrainInputNames()

    for epoch in range(cfg.training.num_epochs):
        print(f'Loading dataset for epoch {epoch + 1}...')
        dataloader = load_dataset(cfg.training, cfg.model.max_seq_length)
        num_batches = len(dataloader)

        training_overview = ("\n"
            f"Number of epochs: {cfg.training.num_epochs}\n"
            f"Batch size: {cfg.training.batch_size}\n"
            f"Number of batches (steps): {num_batches:,}\n")
        print(training_overview)
        
        print(f'Starting training for epoch {epoch + 1}...')
        for i, batch in enumerate(dataloader):
            # No need for optimizer.zero_grad() since the optimizer is in the execution graph

            # Convert (input_ids, token_type_ids) to (input_ids, attn_mask, labels)
            model_inputs = inputs_processor.prepare_inputs_to_train(**batch)

            # Move tensors to RDU and assign their correct name
            _move_tensors_to_rdu(model_inputs, samba_tensor_names)

            # Create a gradient scaling tensor to mask out padding tokens
            grad_scale = inputs_processor.get_gradient_scale(batch['token_type_ids'])

            # Set the gradient scale for the loss. It will be used for the next iteration of backward.
            loss_tensor = model_outputs[0]
            loss_tensor.sn_grad = grad_scale

            # Forward, backward, and optim happen in a single samba.session.run call
            outputs = samba.session.run(model_inputs, model_outputs)

            # Reduce the loss tensor wrt grad scale to report the mean loss
            loss = samba.to_torch(outputs[0]).float()
            loss *= grad_scale.float()
            avg_step_loss = loss.sum().item()
            print(f'Epoch [{epoch+1}/{cfg.training.num_epochs}], Step [{i+1}/{num_batches}], Loss: {avg_step_loss:.4f}')

    print('Finished training.')

    print('Saving checkpoint...')
    save_as_huggingface_checkpoint(model, cfg.checkpoint, cfg.training.output_dir)
    print(f'Checkpoint saved at {cfg.training.output_dir}/')

    print('Saving summary...')
    mz_params_header = "\nThe following are the model params used to train this model using Model Zoo:\n"
    summary_text = training_overview + mz_params_header + cfg.model.model_dump_json()
    save_summary_report(cfg.training.output_dir, 'summary.txt', summary_text)
    print(f'Summary saved at {cfg.training.output_dir}/summary.txt')


if __name__ == '__main__':
    main()
