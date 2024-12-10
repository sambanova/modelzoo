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

import time
from typing import Dict

import hydra
import torch
from accelerate import init_empty_weights
from config.schema import CheckpointConfig, PretrainedModelConfig, RDUTrainingConfig, TrainingConfig
from sambanova_modelzoo.libs.common.arguments import to_pydantic
from sambanova_modelzoo.libs.common.pef_meta import APP_ARGS_KEY, from_pef_meta_dict, to_pef_meta_dict
from sambanova_modelzoo.libs.nlp.core.clm_runtime import PretrainRuntime, SambaPretrainInputNames
from sambanova_modelzoo.libs.nlp.core.clm_tracer import PretrainTracer
from sambanova_modelzoo.libs.nlp.core.token_utils import COMPLETION_TOKEN, PROMPT_TOKEN
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, set_seed
from utils.checkpoint import save_as_huggingface_checkpoint
from utils.dataset import HDF5ParallelLoader, get_dataset_metadata
from utils.reporting import save_per_step_report, save_summary_report

from sambaflow import samba
from sambaflow.samba import SambaTensor
from sambaflow.samba.utils import get_rank, get_world_size, is_main_process
from sambaflow.samba.utils import set_seed as set_samba_seed
from sambaflow.samba.utils import trace_graph
from sambaflow.samba.utils.checkpoint import load_sharded_checkpoint
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

    # Load model from the config. If pretraining, randomly initialized weights will be transferred to the RDU. If
    # finetuning from a checkpoint on disk, load the weights to RDU after tracing.
    model = AutoModelForCausalLM.from_config(config, torch_dtype=model_arch.dtype)

    return model


def load_dataset(training: TrainingConfig, model_seq_length: int, for_eval: bool = False) -> DataLoader:
    """ Load a dataset folder prepared by generative_data_prep """

    dataset_path = str(training.dataset)

    # generative_data_prep embeds some metadata in the dataset folder
    # Use it to confirm compatibility with this model
    dataset_seq_length = get_dataset_metadata(dataset_path)['max_seq_length']
    if dataset_seq_length != model_seq_length:
        raise RuntimeError(
            f"This dataset (seq len {dataset_seq_length}) is not compatible with this model (seq len {model_seq_length})"
        )

    rank = 0 if for_eval else get_rank()
    world_size = 1 if for_eval else get_world_size()

    # ModelZoo LLMs expect data processed using generative_data_prep.
    # Each sample is {'input_ids': List[List[int]], 'token_type_ids': List[List[int]]}
    # For more details on token_type_ids, refer to TODO: article attention docs
    dataloader = HDF5ParallelLoader(dataset_path,
                                    dataset_seq_length,
                                    training.batch_size,
                                    rank,
                                    world_size,
                                    training.seed,
                                    drop_last=True,
                                    train=not for_eval)
    return dataloader


def compile(cfg: RDUTrainingConfig, tracing_inputs: Dict[str, "torch.tensor"], model: PreTrainedModel,
            optimizer: samba.optim.AdamW) -> str:
    """ Compile a PEF to run training on RDU """

    # Get the arguments for the compiler
    compile_dict = cfg.samba_compile.model_dump()

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


# Read args from base_config_rdu_dp.yaml using Hydra and convert them to RDUTrainingConfig pydantic model
@hydra.main(config_path="config", config_name="base_config_rdu_dp", version_base="1.2")
@to_pydantic(RDUTrainingConfig)
def main(cfg: RDUTrainingConfig):
    # Bring up torch.distributed
    torch.distributed.init_process_group(backend='mpi')

    # Read compile time information from the PEF metadata
    cfg = from_pef_meta_dict(cfg) if cfg.command == 'run' else cfg

    if is_main_process():
        print('Running app with arguments:')
        print(cfg.model_dump())

    # Set the seed for reproducibility between runs or workers
    set_seed(cfg.training.seed)
    set_samba_seed(cfg.training.seed)

    print('Loading model...')
    # Load the model for tracing without materializing the weights.
    with init_empty_weights():
        model = load_model(cfg.checkpoint, cfg.model)
    samba.from_torch_model_(model)  # important! Move model to samba runtime

    print('Loading optimizer...')
    import sambaflow.samba.optim as optim  # important! Load the optimizer from samba instead

    # if model params are empty/meta tensors, the optim state tensors will also be empty/meta tensors
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
        # transfer_device=False so that tensor data is not automatically transferred from CPU to RDU. If pretraining,
        # weight data will be initialized and transferred to RDU later, and if finetuning, weight data will be loaded
        # from a checkpoint directly onto the RDU through load_sharded_checkpoint
        model_outputs = trace_graph(model,
                                    tracing_inputs,
                                    optimizer,
                                    pef=cfg.samba_run.pef,
                                    loss_indices=[0],
                                    data_parallel_mode='normal',
                                    transfer_device=False)
        if cfg.checkpoint.model_name_or_path is not None:
            print('Transferring tensors that have data...')
            traced_tensors_with_data = [t for t in samba.session._traced_tensors() if t.has_data]
            for t in traced_tensors_with_data:
                t.rdu()

            print('Materializing optimizer tensors onto the RDU...')
            for optim_tensor in samba.session.optim_dict.values():
                optim_tensor.rdu()

            # load checkpoint directly onto the RDU without keeping a copy of the data on the host, in order to avoid
            # OOM
            print('Loading sharded checkpoint...')
            load_sharded_checkpoint(model, cfg.checkpoint.model_name_or_path, ignore_dtype_mismatch=True)
        else:
            print('Reinitializing model and transferring weights to RDU...')
            # need to reinitialize model to replace meta tensors with proper values
            model = load_model(cfg.checkpoint, cfg.model)
            samba.from_torch_model_(model)
            samba.session.to_device()
    else:
        raise ValueError('Only compile and run commands are supported')

    # ModelZoo's PretrainRuntime will be used in the training loop to
    # create attention masks and labels from the input data
    inputs_processor = PretrainRuntime(cfg.model.max_seq_length)

    # Keep track of per step metrics for reporting
    tokens_per_step = []
    time_per_step = []
    loss_per_step = []
    lr_per_step = []

    # When tensors are moved to RDU in the traning loop, they need to be assigned a name.
    # Each tensor has a name in the execution graph decided during compile time.
    # The input tensors need to match their unique names at runtime.
    samba_tensor_names = SambaPretrainInputNames()

    break_training = False

    start_train_time = time.time()
    for epoch in range(cfg.training.num_epochs):
        if break_training:
            break

        print(f'Loading dataset for epoch {epoch + 1}...')
        dataloader = load_dataset(cfg.training, cfg.model.max_seq_length)
        num_batches = len(dataloader)
        world_size = get_world_size()

        training_overview = ("\n"
                             f"Number of epochs: {cfg.training.num_epochs}\n"
                             f"Per worker batch size: {cfg.training.batch_size}\n"
                             f"Per worker number of batches (steps): {num_batches:,}\n"
                             f"Number of DP workers: {world_size}\n")
        if is_main_process():
            print(training_overview)

        print(f'Starting training for epoch {epoch + 1}...')
        for i, batch in enumerate(dataloader):
            start_step_time = time.time()
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
            outputs = samba.session.run(model_inputs, loss_tensor, data_parallel=True, reduce_on_rdu=True)

            # Reduce the loss tensor wrt grad scale between workers to report the mean loss
            loss = samba.to_torch(outputs[0]).float()
            loss *= grad_scale.float()
            loss = loss.sum()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss)
            avg_step_loss = loss / world_size
            end_step_time = time.time()

            if is_main_process():
                num_prompt_tokens = (batch['token_type_ids'] == PROMPT_TOKEN).sum()
                num_completion_tokens = (batch['token_type_ids'] == COMPLETION_TOKEN).sum()
                tokens_this_step = num_prompt_tokens + num_completion_tokens

                print(
                    f'DP: {world_size}-way, '
                    f'Epoch [{epoch+1}/{cfg.training.num_epochs}], '
                    f'Step [{i+1}/{num_batches}], '
                    f'Loss: {avg_step_loss:.4f}, '
                    f'LR: {cfg.training.learning_rate}, '
                    f'Num tokens this step: {tokens_this_step}',
                    flush=True)

                tokens_per_step.append(tokens_this_step)
                loss_per_step.append(avg_step_loss)
                lr_per_step.append(cfg.training.learning_rate)
                time_per_step.append(end_step_time - start_step_time)

            if i + 1 == cfg.training.end_early_at_step:
                break_training = True
                break

        total_train_time = (time.time() - start_train_time)

        # Evaluate loss/perplexity on the dev set
        if is_main_process() and cfg.training.evaluate:
            print('Running evaluation...')
            loss_per_batch = []
            eval_dataloader = load_dataset(cfg.training, cfg.model.max_seq_length, for_eval=True)
            for batch in tqdm(eval_dataloader):
                # All steps are the same as training, but we do not run backward or optim step
                model_inputs = inputs_processor.prepare_inputs_to_train(**batch)
                _move_tensors_to_rdu(model_inputs, samba_tensor_names)
                model_outputs[0].sn_grad = inputs_processor.get_gradient_scale(batch['token_type_ids'])

                # Only run forward
                outputs = samba.session.run(model_inputs, model_outputs[0], section_types=['fwd'])

                grad_scale = inputs_processor.get_gradient_scale(batch['token_type_ids'])
                loss = (samba.to_torch(outputs[0]).float() * grad_scale.float()).sum()
                loss_per_batch.append(loss)

            mean_loss = sum(loss_per_batch) / len(loss_per_batch)
            perplexity = torch.exp(torch.tensor([mean_loss])).item()
            print(f'Evaluation loss: {mean_loss}, Perplexity: {perplexity}')

        # Wait for all processes (in case rank 0 is evaluating) before next epoch
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    print('Finished training.')

    if is_main_process():
        print('Saving checkpoint...')
        save_as_huggingface_checkpoint(model, cfg.checkpoint, cfg.training.output_dir)
        print(f'Checkpoint saved at {cfg.training.output_dir}/')

        print('Saving summary...')
        total_tokens_seen = f'Total tokens seen: {sum(tokens_per_step)}\n'
        tokens_per_second = f'Tokens per second: {(sum(tokens_per_step) / total_train_time):.4f}\n' if total_train_time else ""
        average_time_per_step = f'Average time per step: {(sum(time_per_step) / len(time_per_step)):.4f}s' if time_per_step else ""
        mz_params_header = "\nThe following are the model params used to train this model using Model Zoo:\n"

        # Create a summary report with the training metrics
        summary_text = training_overview + total_tokens_seen + tokens_per_second + average_time_per_step
        # Add model params to the report
        summary_text += mz_params_header + cfg.model.model_dump_json()

        save_summary_report(cfg.training.output_dir, 'summary.txt', summary_text)
        print(f'Summary saved at {cfg.training.output_dir}/summary.txt')

        print('Saving metrics...')
        metrics = {
            'Tokens in Step': tokens_per_step,
            'Step Loss': loss_per_step,
            'Learning Rate': lr_per_step,
            'Time per Step': time_per_step
        }
        save_per_step_report(cfg.training.output_dir, 'per_step_metrics.csv', metrics)
        print(f'Metrics saved at {cfg.training.output_dir}/per_step_metrics.csv')


if __name__ == '__main__':
    main()
