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

from typing import Tuple

import torch

from sambaflow.samba import SambaTensor
from sambaflow.samba.directives import op_fusion
from sambaflow.samba.functional.stir import sn_imm, sn_iteridx, sn_select, sn_zipmapreduce
from sambaflow.samba.utils import SNType


class TokenGenInit(torch.nn.Module):
    """
    Initializes inputs to the token generation process.

    This module takes in the last token index, attention mask, and generated index,
    and returns the updated attention mask and generated index. The update corrects the attention mask 
    based on the last token index and resets the generated index. 

    Args:
        seq_length (int): The length of the sequence.
        use_plugin_heuristics (bool, optional): Whether to use plugin heuristics. Defaults to False.
        run_early_tp (bool, optional): Whether to run early tensor parallelism. Defaults to False.
    """
    def __init__(self, seq_length: int, use_plugin_heuristics: bool = False, run_early_tp: bool = False):
        super().__init__()
        self.seq_length = seq_length
        self.use_plugin_heuristics = use_plugin_heuristics
        self.run_early_tp = run_early_tp

    def forward(
            self,
            last_token_index: torch.Tensor,
            attention_mask: torch.Tensor,
            generated_index: torch.Tensor,
    ) -> Tuple[SambaTensor, SambaTensor]:
        """
            Performs the forward pass of the token generation initialization.

            Args:
                last_token_index (torch.Tensor): The index of the last token.
                attention_mask (torch.Tensor): The attention mask to be used by the token generation graph.
                generated_index (torch.Tensor): The generated index.

            Returns:
                Tuple[SambaTensor, SambaTensor]: A tuple containing the updated attention mask and generated index.
        """
        if self.run_early_tp == True:
            heuristic = "kPostprocess_SPMD"
            plugins = ["libSPMD_PostprocessPlugins.so"]
        else:
            heuristic = {"distribution": "kPostprocess", "tiling": "kPostprocess", "mapping": "kPostprocess"}
            plugins = ["libPostprocessPlugins.so"]
        with op_fusion(func_name=f"token_gen_init",
                       heuristic=heuristic,
                       plugins=plugins,
                       user_annotated=self.use_plugin_heuristics):

            def create_attention_mask_lambda(attrs, last_index, prev_mask):
                dim_l = sn_iteridx(dim=1, attrs=attrs, dtype=SNType.INT32)
                result = sn_select(dim_l <= last_index, prev_mask, sn_imm(0, SNType.INT32))
                return result

            assert len(last_token_index.shape) == 2

            def create_generated_index(attrs, original_index):
                return original_index * sn_imm(0, SNType.INT32)

            attention_mask = sn_zipmapreduce(create_attention_mask_lambda, [last_token_index, attention_mask])
            generated_index = sn_zipmapreduce(create_generated_index, [generated_index])

        return (attention_mask, generated_index)
