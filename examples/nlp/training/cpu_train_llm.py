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
This file demonstrates how to do the following in a CPU environment:
  1. Load an LLM
        - from a checkpoint
        - from a config
  2. Load a dataset
        - that's been processed using generative_data_prep
  3. Train an LLM
        - without using HF trainer
  4. Save a checkpoint

It is for demonstrative purposes only and not intended to be used for real training.
It follows the same structure as and shares much of its code with rdu_train_llm.py.
"""

import hydra
from config.schema import CheckpointConfig, CPUTrainingConfig, PretrainedModelConfig, TrainingConfig
from sambanova_modelzoo.libs.common.arguments import to_pydantic
from sambanova_modelzoo.libs.nlp.core.clm_runtime import PretrainRuntime
from sambanova_modelzoo.models.configuration_transformer import ConfigurationTransformer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from utils.checkpoint import save_as_huggingface_checkpoint
from utils.dataset import HDF5ParallelLoader, get_dataset_metadata
from utils.reporting import save_summary_report


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
    dataloader = HDF5ParallelLoader(dataset_path,
                                    dataset_seq_length,
                                    local_batch_size=training.batch_size,
                                    drop_last=True)
    return dataloader


# Read args from base_config_cpu.yaml using Hydra and convert them to CPUTrainingConfig pydantic model
@hydra.main(config_path="config", config_name="base_config_cpu", version_base="1.2")
@to_pydantic(CPUTrainingConfig)
def main(cfg: CPUTrainingConfig):

    print('Running app with arguments:')
    print(cfg.model_dump())

    print('Loading model...')
    model = load_model(cfg.checkpoint, cfg.model)

    print('Loading optimizer...')
    import torch.optim as optim
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    # NOTE: In the base model source from huggingface, the model computes the loss
    # So there's no need to define a loss function here

    # ModelZoo's PretrainRuntime will be used in the training loop to
    # create attention masks and labels from the input data
    inputs_processor = PretrainRuntime(cfg.model.max_seq_length)

    break_training = False

    for epoch in range(cfg.training.num_epochs):
        if break_training:
            break

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
            optimizer.zero_grad()

            # Convert (input_ids, token_type_ids) to (input_ids, attn_mask, labels)
            model_inputs = inputs_processor.prepare_inputs_to_train(**batch)

            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss

            # Create a gradient scaling tensor to mask out padding tokens
            grad_scale = inputs_processor.get_gradient_scale(batch['token_type_ids'])

            # Backward pass
            loss.backward(grad_scale)
            optimizer.step()

            # Reduce the loss tensor wrt grad scale to report the mean loss
            loss *= grad_scale.float()
            avg_step_loss = loss.sum().item()
            print(f'Epoch [{epoch+1}/{cfg.training.num_epochs}], Step [{i+1}/{num_batches}], Loss: {avg_step_loss:.4f}')

            if i + 1 == cfg.training.end_early_at_step:
                break_training = True
                break

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
