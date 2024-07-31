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

from sambanova_modelzoo.libs.nlp.core.token_utils import ARTICLE_SEP_TOKEN_TYPE_ID, PADDING_TOKEN_TYPE_ID


def get_collapsed_article_attention_mask(target_token_type_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Article attention mask causes tokens to only attend to other tokens in the same article.                                                                         
                                                                                                                        
    The attention mask is defined by having a 1 at row i, column j if you should attend to token j when learning to
    generate token i+i. Otherwise it will be 0's.                                                                                           

    Example sequence with two articles:                                                                                 
    Sequence: [art1, art1, SEP, art2, art2,  SEP]                                                                        
    token_type_ids = [0,1,3,0,1,3]                                                                                      
    target_token_type_ids = [1,3,0,1,3,2]                                                                               
    
    with 0: prompt token, 1: completion token, 2: padding_token, 3: eos_token
    more info: https://github.sambanovasystems.com/SambaNova/generative_data_prep#output-1

    The attention mask generated on chip will be:
    1 0 0 0 0 0                                                                                                         
    1 1 0 0 0 0                                                                                                         
    0 0 0 0 0 0                                                                                                         
    0 0 0 1 0 0                                                                                                         
    0 0 0 1 1 0                                                                                                         
    0 0 0 0 0 0 

    Collapsed article attention mask is a pair of tensor used to construct the above full  article attention mask 
    on chip.  We do not want to directly construct the full size article attention mask and transfer it to the chip
    because the mask would be of O(seq_len^2) which can be bandwidth heavy.

    The reference of on-chip expansion can be found in models.custom_ops.create_article_attention custom ops.

    Args:
        target_token_type_ids: [batch_size x sequence_length] The token type ids of the gold tokens the model must predict.

    Returns:
        The article attention mask represented by two collapsed tensors.  The actual article attention mask
        is of size [batch_size, 1, sequence_length, sequence_length] generated on chip. The two collapsed tensors are 
        each of size [batch_size, 1, 1, sequence_length].
    """
    attention_mask = torch.zeros(target_token_type_ids.shape)
    end_token_ids = {ARTICLE_SEP_TOKEN_TYPE_ID, PADDING_TOKEN_TYPE_ID}
    for sequence_idx, sequence in enumerate(target_token_type_ids):
        num_articles = 0
        start_index = 0
        # for each sample in the batch, the collapsed attention mask looks like:
        # [1, 1, .... 1, 0, 2, 2, ... 2, 0, ... n, n ..... n], assuming there are n articles in the sequence.
        # Each of the n articles are separated by a 0.
        for token_idx, token_type_id in enumerate(sequence):
            if start_index is not None and token_type_id.item() in end_token_ids:
                num_articles += 1
                # Do not include padding tokens into mask consideration. We do want to learn SEP
                end_index = token_idx if token_type_id == PADDING_TOKEN_TYPE_ID else token_idx + 1
                attention_mask[sequence_idx][start_index:end_index] = num_articles
                start_index = None
            elif start_index is None and token_type_id not in end_token_ids:
                start_index = token_idx + 1

    # do an unsqueeze to convert our attention mask from [bs, ss] to [bs, 1, ss].  This is required so that we
    # can create the transposed attention mask which is of size [bs, ss, 1]
    attention_mask = attention_mask.unsqueeze(1)
    attention_mask_t = attention_mask.clone().transpose(-1, -2)
    attention_mask_t[attention_mask_t == 0] = -1
    # do an extra unsqueeze here so we don't need to reshape on chip.  This ensures that the mask gets broadcasted
    # across the head dimension.
    attention_mask, attention_mask_t = (mask.unsqueeze(1) for mask in [attention_mask, attention_mask_t])
    return attention_mask, attention_mask_t
