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

import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import coe_api
import torch
from accelerate import init_empty_weights
from sambanova_modelzoo.libs.common.pretrained_model_schema import PretrainedModelConfig
from sambanova_modelzoo.libs.nlp.core.clm_runtime import CachedInferenceRuntime
from sambanova_modelzoo.models.utils import load_model_from_pretrained
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .base import PipelineBase


class ReturnType(enum.Enum):
    TENSORS = enum.auto()
    NEW_TEXT = enum.auto()
    FULL_TEXT = enum.auto()


class FrameworkType(str, enum.Enum):
    PYTORCH = "pt"

    def __str__(self):
        return self.value


class TextGenerationPipeline(PipelineBase):
    """
    A pipeline for (text-only) language model text generation tasks.

    This class extends PipelineBase to implement specific functionality
    for text generation models, including preprocessing of input text,
    model inference, and postprocessing of generated text.

    Attributes:
        framework (str): The deep learning framework used, set to FrameworkType.TORCH.
    """

    framework = FrameworkType.PYTORCH

    def _sanitize_parameters(self, **pipeline_parameters) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Sanitize and separate user provided parameters for each of the different pipeline stages:
            preprocess -> forward -> postprocess

        Args:
            **pipeline_parameters: Keyword arguments containing all pipeline parameters.

        Returns:
            tuple: A tuple containing three dictionaries:
                - preprocessing_params (Dict[str, Any]): Parameters for the preprocess method.
                - forward_params (Dict[str, Any]): Parameters for the forward method.
                - postprocessing_params (Dict[str, Any]): Parameters for the postprocess method.
        """
        preprocessing_kwargs = ['prefix', 'add_special_tokens', 'truncation', 'padding', 'max_length']
        preprocessing_params = {
            key: pipeline_parameters[key]
            for key in preprocessing_kwargs if key in pipeline_parameters
        }

        forward_kwargs = ['output_logits', 'max_length', 'max_new_tokens']
        forward_params = {key: pipeline_parameters[key] for key in forward_kwargs if key in pipeline_parameters}

        postprocessing_kwargs = ['output_logits', 'return_type', 'clean_up_tokenization_spaces']
        postprocessing_params = {
            key: pipeline_parameters[key]
            for key in postprocessing_kwargs if key in pipeline_parameters
        }

        return preprocessing_params, forward_params, postprocessing_params

    def preprocess(
            self,
            prompt_text: Union[str, List[str]],
            prefix: Optional[str] = "",
            add_special_tokens: Optional[bool] = None,
            truncation: Optional[bool] = None,
            padding: Optional[bool] = None,
            max_length: Optional[int] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        """
        Preprocess the input text for the text generation model.

        Args:
            prompt_text (str or list): The input text or list of texts to be processed.
            prefix (str, optional): A prefix to be added to each input text. Defaults to "".
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to None.
            truncation (bool, optional): Whether to truncate sequences. Defaults to None.
            padding (bool or str, optional): Padding strategy. Defaults to None.
            max_length (int, optional): Maximum length of processed sequences. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing preprocessed inputs, including tokenized input_ids and prompt_text.
        """
        tokenizer_kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        tokenizer_kwargs = {key: value for key, value in tokenizer_kwargs.items() if value is not None}
        if prefix:
            prompt_text = [prefix + _txt for _txt in prompt_text]
        inputs = self.tokenizer(prompt_text, return_tensors=self.framework, **tokenizer_kwargs)

        inputs["prompt_text"] = prompt_text

        return inputs

    def postprocess(self,
                    model_outputs: Dict[str, Union[str, torch.Tensor]],
                    return_type: Optional[ReturnType] = ReturnType.FULL_TEXT,
                    clean_up_tokenization_spaces: Optional[bool] = True,
                    output_logits: Optional[bool] = False) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """
        Postprocess the model outputs to generate final text and additional information.

        Args:
            model_outputs (Dict[str, Union[str, torch.Tensor]]): The raw outputs from the model.
            return_type (ReturnType, optional): The type of text to return. Defaults to ReturnType.FULL_TEXT.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Defaults to True.
            output_logits (bool, optional): Whether to include logits in the output. Defaults to False.

        Returns:
            List[Dict[str, Union[str, torch.Tensor]]]: A list of dictionaries, each containing generated text and additional information for each input.
        """
        input_ids = model_outputs["input_ids"]
        input_length = input_ids.shape[-1]
        generated_ids = model_outputs["sequences"][:, input_length:]
        prompts = model_outputs["prompt_text"]

        completions = self.tokenizer.batch_decode(generated_ids,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=clean_up_tokenization_spaces)

        if output_logits:
            logits = torch.stack(model_outputs['logits'], dim=1)
            logits = list(torch.split(logits, 1, dim=0))

        batch_response = []
        for batch_row, (prompt, completion) in enumerate(zip(prompts, completions)):
            prompt_tokens = input_ids[batch_row].tolist()
            # Remove padding tokens from prompt
            prompt_tokens = self._remove_padding(prompt_tokens)

            tokens = generated_ids[batch_row].tolist()
            stop_reason = self._determine_stop_reason(tokens)

            record = self._create_output_record(return_type, prompt, completion, prompt_tokens, generated_ids,
                                                stop_reason)

            if output_logits:
                record["logits"] = logits[batch_row].squeeze(dim=0)

            batch_response.append(record)

        return batch_response

    def _remove_padding(self, prompt_tokens: List[int]) -> List[int]:
        """
        Remove padding tokens from the prompt.

        Args:
            prompt_tokens (List[int]): List of token IDs.

        Returns:
            List[int]: List of token IDs with padding removed.
        """
        # note: this method of removing pad tokens from prompt_tokens uses more code
        # than a filter with a lamba, but is dramatically faster for large prompts
        if len(prompt_tokens) != 0 and prompt_tokens[0] == self.tokenizer.pad_token_id:
            # remove prompt padding that has been added on the left
            for first_nonpad in range(len(prompt_tokens)):
                if prompt_tokens[first_nonpad] != self.tokenizer.pad_token_id:
                    break
            # truncate the prompt before the first nonpad token
            prompt_tokens = prompt_tokens[first_nonpad:]
        if len(prompt_tokens) != 0 and prompt_tokens[-1] == self.tokenizer.pad_token_id:
            # remove prompt padding that has been added on the right
            for last_non_pad in range(len(prompt_tokens) - 1, -1, -1):
                if prompt_tokens[last_non_pad] != self.tokenizer.pad_token_id:
                    break
            # truncate the prompt after the last non-pad token
            prompt_tokens = prompt_tokens[:last_non_pad + 1]
        if self.tokenizer.pad_token_id in prompt_tokens:
            # found a pad token embedded in the prompt, so need to clean it the slow way
            prompt_tokens = list(filter(lambda _token: _token != self.tokenizer.pad_token_id, prompt_tokens))
        return prompt_tokens

    def _determine_stop_reason(self, tokens: List[int]) -> str:
        """
        Determine the reason for stopping text generation.

        Args:
            tokens (List[int]): List of generated token IDs.

        Returns:
            str: The reason for stopping ('end_of_text' or 'max_len_reached').
        """
        if self.tokenizer.decode([tokens[-1]]) == self.tokenizer.eos_token:
            return 'end_of_text'
        return 'max_len_reached'

    def _create_output_record(self, return_type: ReturnType, prompt: str, completion: str, prompt_tokens: List[int],
                              generated_ids: torch.Tensor, stop_reason: str) -> Dict[str, str]:
        """
        Create the output record based on the return type.

        Args:
            return_type (ReturnType): The type of text to return.
            prompt (str): The original prompt.
            completion (str): The generated completion.
            prompt_tokens (List[int]): List of prompt token IDs.
            generated_ids (torch.Tensor): Tensor of generated token IDs.
            stop_reason (str): The reason for stopping generation.

        Returns:
            dict: A dictionary containing the output record.
        """
        if return_type == ReturnType.TENSORS:
            return {"generated_token_ids": generated_ids, "stop_reason": stop_reason}
        elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            return_text = completion if return_type == ReturnType.NEW_TEXT else prompt + completion
            return {"generated_text": return_text, "stop_reason": stop_reason}
        return {}


class TorchMonolithLLMPipeline(TextGenerationPipeline):
    """
    A pipeline for text generation using a single language model (hence the name Monolith).

    This class extends TextGenerationPipeline to implement specific functionality
    for loading and running large language models from Hugging Face checkpoints 
    specifically for torch native devices such as cpu and cuda devices.cpu and cuda devices.

    Attributes:
        device (str): The device to run the model on ('cpu' or 'cuda').
        model_name_or_path (str): Path to the Hugging Face model checkpoint.
        sn_model_config (PretrainedModelConfig, optional): SN Modelzoo configuration for the model.
        original_config_overrides (dict): Configs that override the original model configurations.
        tokenizer (PretrainedTokenizer): Tokenizer for the model.
        rng (torch.Generator): Random number generator for reproducibility.
    """
    def __init__(self,
                 model_name_or_path: str,
                 sn_model_config: Optional[PretrainedModelConfig] = None,
                 original_config_overrides: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = 42,
                 device: Optional[str] = "cpu") -> None:
        """
        Initialize the TorchMonolithLLMPipeline.

        Args:
            model_name_or_path (str): Path to the Hugging Face model checkpoint.
            sn_model_config (PretrainedModelConfig, optional): Configuration for the model. Defaults to None.
                If not provided, the HF version of the model will be instantiated.
            original_config_overrides: Configs that override the original model configurations.
            seed (int, optional): Seed for the random number generator. Defaults to 42.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to "cpu".

        Raises:
            AssertionError: If the device is not 'cpu' or 'cuda'.
        """
        assert device in ["cpu", "cuda"]
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.sn_model_config = sn_model_config
        self.original_config_overrides = original_config_overrides

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def _init_model(self) -> None:
        # Instantiate from samba modelzoo if a sn_model_config is provided
        if self.sn_model_config:
            self.model = load_model_from_pretrained(self.model_name_or_path, self.sn_model_config,
                                                    self.original_config_overrides)
        # Else just instantiate a HF model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        # Ensure the model is on CPU and in evaluation mode
        self.model = self.model.to(self.device).eval()

    def forward(self, inputs: Dict[str, Union[str, torch.Tensor]], output_logits: Optional[bool] = False,
                **kwargs) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Perform the forward pass of the model on cpu or cuda device.

        This method initializes the model, generates text based on the input,
        and manages memory by deleting the model after generation.

        Args:
            inputs (Dict[str, Union[str, torch.Tensor]]): A dictionary containing the input data, including 'prompt_text' and 'input_ids'.
            output_logits (bool, optional): Whether to output logits. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model's generate method.

        Returns:
            Dict[str, Union[str, torch.Tensor]]: A dictionary containing the model outputs, including generated sequences,
                  prompt text, and input IDs. If output_logits is True, it also includes the logits.

        Note:
            This method deletes the model from memory after generation to free up resources.
        """
        self._init_model()
        _prompt = inputs.pop("prompt_text")
        with torch.no_grad():
            model_outputs = self.model.generate(**inputs,
                                                return_dict_in_generate=True,
                                                output_logits=output_logits,
                                                **kwargs)
        model_outputs["prompt_text"] = _prompt
        model_outputs["input_ids"] = inputs["input_ids"]
        del self.model
        return model_outputs


def _load_checkpoint(coe_app: coe_api.MLApp,
                     path: str,
                     max_seq_length: int = 4096,
                     static_sequence_lengths=[],
                     call_graph_length=None) -> Tuple:
    """Internal util to manage coe_api.MLapp (to eventually be refactored)"""
    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = 'right'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    sliding_window = getattr(config, "sliding_window", None)

    # Warning: this script uses the CachedInferenceRuntime base class. This will need to be updated if a model
    # subclasses CachedInferenceRuntime and overrides any functions that are used here
    inputs_processor = CachedInferenceRuntime(max_seq_length,
                                              tokenizer.pad_token_id,
                                              sliding_window=sliding_window,
                                              runner="samba")

    coe_checkpoint = coe_app.load_checkpoint(path)

    def model_rdu_step(_self, *input, **kwargs):
        use_cache = bool(_self.model_rdu_step_run_count > 0)

        input_id_length = kwargs['input_ids'].shape[1]

        if use_cache:
            inputs = inputs_processor.preprocess_inputs_for_token_gen(kwargs['input_ids'][:, -1:])
        else:
            if static_sequence_lengths:
                if not call_graph_length:
                    static_seq_len = min([s for s in static_sequence_lengths if s >= input_id_length])
                else:
                    static_seq_len = call_graph_length
            else:
                static_seq_len = max_seq_length
            inputs = inputs_processor.preprocess_inputs_for_cache_gen(kwargs['input_ids'], kwargs['attention_mask'],
                                                                      static_seq_len)

        if use_cache:
            graph_name = 'model_cache_' + str(max_seq_length) + '_fwd'
        else:
            graph_name = 'model_nocache_' + str(static_seq_len) + '_fwd'
        out = coe_app.run(coe_checkpoint, inputs, graph_name, reuse_outputs=True)
        output_logits = out[0]
        _self.model_rdu_step_run_count += 1

        logits = output_logits[:, :input_id_length, :].float()
        return CausalLMOutputWithCrossAttentions(loss=None, logits=logits)

    # A class can only have one implementation of a function, but we
    # need to provide different closures for each loaded checkpoint.
    #
    # Solution: Forward the class __call__ to the instance __call__ if present,
    #           and put the closure on the instance
    orig_call = model.__class__.__call__

    def forward_call_to_instance(_self, *input, **kwargs):
        instance_call = getattr(_self, "__call__", None)
        if instance_call:
            return instance_call(_self, *input, **kwargs)
        else:
            return orig_call(_self, *input, **kwargs)

    model.model_rdu_step_run_count = 0
    model.__class__.__call__ = forward_call_to_instance
    model.__call__ = model_rdu_step

    return model, tokenizer


class RDUMonolithLLMPipeline(TextGenerationPipeline):
    """
    A pipeline for text generation using a single language model (hence the name Monolith).
    This is specially implemented to run the model on SambaNova's Reconfigurable Dataflow Unit (RDU).

    This class extends TextGenerationPipeline to implement specific functionality
    for loading and running large language models on SambaNova's deep learning accelerator.
    It leverages the coe_api framework built by SambaNova Systems to allow performant
    execution on RDUs.

    Attributes:
        device (str): Set to "rdu" to indicate execution on SambaNova's RDU.
        pef (str): Path to the Portable Executable Format (PEF) file, a special
                   executable format for compiled models on the RDU.
        batch_size (int): Batch size for model inference.
        copy_pef (str): Path to a supporting executable for pipelining weight loading to HBM.
        max_pef_len (int): Maximum sequence length that the PEF has been compiled to support.
        static_seq_lens (List[int]): Sequence lengths compiled into the cache generation graph.
        call_graph_len (int): Override flag for choosing a specific length on the cache generation graph.
        model_name_or_path (str): Path to the Hugging Face model checkpoint.
        ml_app (coe_api.MLApp): SambaNova's ML application object for RDU execution.
        model: The loaded language model.
        tokenizer: The tokenizer for the loaded model.
    """
    def __init__(self,
                 model_name_or_path: str,
                 pef: str,
                 max_pef_len: int,
                 batch_size: Optional[int] = 1,
                 static_seq_lens: Optional[List[int]] = [],
                 call_graph_len: Optional[int] = None,
                 copy_pef: Optional[str] = None,
                 target_runtime_version="LATEST") -> None:
        self.device = "rdu"
        self.pef = pef
        self.batch_size = batch_size
        self.copy_pef = copy_pef
        self.max_pef_len = max_pef_len
        self.static_seq_lens = static_seq_lens
        self.model_name_or_path = model_name_or_path
        self.set_call_graph_len(call_graph_len)
        coe_api.set_target_runtime_version(target_runtime_version)

        self.ml_app = coe_api.MLApp(self.pef, copy_pef_path=self.copy_pef)
        self.model, self.tokenizer = _load_checkpoint(self.ml_app, self.model_name_or_path, self.max_pef_len,
                                                      self.static_seq_lens, self.call_graph_len)

    def forward(self, inputs: Dict[str, Union[str, torch.Tensor]], output_logits: Optional[bool] = False,
                **kwargs) -> Dict:
        """
        Perform the forward pass of the model on RDU.

        This method initializes the model, generates text based on the input,
        and manages memory by deleting the model after generation.

        Args:
            inputs (Dict[str, Union[str, torch.Tensor]]): A dictionary containing the input data, including 'prompt_text' and 'input_ids'.
            output_logits (bool, optional): Whether to output logits. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model's generate method.

        Returns:
            Dict: A dictionary containing the model outputs, including generated sequences,
                  prompt text, and input IDs. If output_logits is True, it also includes the logits.
        """
        _prompt = inputs.pop("prompt_text")
        coe_api.start_profile()
        assert len(inputs["input_ids"]) == self.batch_size
        model_outputs = self.model.generate(input_ids=inputs['input_ids'],
                                            attention_mask=inputs['attention_mask'],
                                            output_logits=output_logits,
                                            return_dict_in_generate=True,
                                            **kwargs)
        model_outputs["prompt_text"] = _prompt
        model_outputs["input_ids"] = inputs["input_ids"]
        coe_api.end_profile()
        return model_outputs

    def set_call_graph_len(self, call_graph_len: Optional[int] = None) -> None:
        """
        Set the token generation graph to be called in a multi-graph PEF.

        This method sets the call_graph_len attribute, which is used to override
        the choice of a specific length on the cache generation graph.

        Args:
            call_graph_len (Optional[int]): The desired call graph length. Defaults to None.

        Raises:
            AssertionError: If the provided call_graph_len is not in the list of
                            static_seq_lens compiled into the cache generation graph.

        Note:
            The call_graph_len must be one of the values in the static_seq_lens list,
            which represents the sequence lengths compiled into the cache generation graph.
        """
        if (call_graph_len is not None) and (call_graph_len not in self.static_seq_lens):
            raise ValueError(f"call_graph_len must be one of {self.static_seq_lens}")
        self.call_graph_len = call_graph_len
