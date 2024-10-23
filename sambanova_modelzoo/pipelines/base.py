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


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class PipelineBase(ABC):
    """
    Base class that governs user-facing behavior of the pipeline class.
    This class is analogous to Hugging Face's `Pipeline` class and provides
    a structure for implementing machine learning pipelines with preprocessing,
    forward pass, and postprocessing steps.
    """
    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        Sanitize and resolve parameters for the pipeline steps.
        This method should return three dictionaries containing the resolved parameters
        used by the `preprocess`, `forward`, and `postprocess` methods respectively.
        Args:
            **pipeline_parameters: Keyword arguments containing pipeline parameters.
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: A tuple containing three dictionaries:
                - preprocess_params: Parameters for the preprocess method.
                - forward_params: Parameters for the forward method.
                - postprocess_params: Parameters for the postprocess method.
        Note:
            - Do not fill dictionaries if the caller didn't specify kwargs.
            - This method is not meant to be called directly. It will be automatically
              called by `__init__` and `predict` methods.
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Preprocess the input data before the forward pass.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            Dict[str, Any]: Preprocessed input ready for the model's forward pass.
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Perform the forward pass of the model.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            Dict[str, Any]: Raw output from the model's forward pass.
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """
        Postprocess the model's output.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            Dict[str, Any]: Final processed output of the pipeline.
        """
        raise NotImplementedError()

    def predict(self, inputs: Union[str, List[str]], *args, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run the full pipeline on the given input(s).
        Args:
            inputs: Either a single input string or a list of input strings.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments for pipeline parameters.
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Output of the pipeline for single or multiple inputs.
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        if isinstance(inputs, str):
            inputs = [inputs]
        multi_batch = isinstance(inputs[0], list)
        if multi_batch:
            return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_single(self, inputs: List[str], preprocess_params: Dict[str, Any], forward_params: Dict[str, Any],
                   postprocess_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the pipeline on a single batch of inputs.
        Args:
            inputs: A list of input strings.
            preprocess_params: Parameters for preprocessing.
            forward_params: Parameters for the forward pass.
            postprocess_params: Parameters for postprocessing.
        Returns:
            Dict: The final output of the pipeline for the given inputs.
        """
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def run_multi(self, inputs: List[List[str]], preprocess_params: Dict[str, Any], forward_params: Dict[str, Any],
                  postprocess_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run the pipeline on multiple batches of inputs.
        Args:
            inputs: A list of lists, where each inner list contains input strings.
            preprocess_params: Parameters for preprocessing.
            forward_params: Parameters for the forward pass.
            postprocess_params: Parameters for postprocessing.
        Returns:
            List[Dict]: A list of pipeline outputs, one for each batch of inputs.
        """
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]
