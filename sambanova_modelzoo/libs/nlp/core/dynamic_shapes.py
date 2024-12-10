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

from dataclasses import dataclass
from typing import Dict, Optional

import sambaflow.samba as samba

"""
This file contains the code for handling dynamic shapes in ModelZoo.
"""
@dataclass
class DynamicShapes:
    """
    A mapping from human-readable dimension names to the dynamic dimension name used in Modelzoo that is registered with
    :class:`SambaSession`.

    Attributes:
        batch_size: human-readable name of the batch_size dynamic dimension. If ``None``, there is no dynamic batch_size
            dimension
        sequence_length_cache_gen: mapping from each cache_gen graph's sequence length to the human-readable name of
            its dynamic dimension. If it is an empty dictionary, there are no dynamic sequence_length dimensions for the
            cache_gen graph
        sequence_length_token_gen: human-readable name of the token_gen graph's sequence length dimension. If ``None``, there is no dynamic sequence_length
            dimension for the token_gen graph
    """
    batch_size: Optional[str]
    sequence_length_cache_gen: Dict[int, str]
    sequence_length_token_gen: Optional[str]

    @staticmethod
    def create() -> 'DynamicShapes':
        """Creates an empty DynamicShapes instance

        Returns:
            an empty DynamicShapes instance
        """
        return DynamicShapes(batch_size=None, sequence_length_cache_gen={}, sequence_length_token_gen=None)


class DynamicShapesBatchSizeError(RuntimeError):
    """Raised when multiple dynamic dims for batch size are specified."""


class DynamicShapesCacheGenDuplicateError(RuntimeError):
    """Raised when multiple dynamic dims for the cache_gen graph's sequence length with the same max value are
    specified. However, multiple dynamic dims for the cache_gen graph's sequence length are allowed as long as the max
    values are different."""


class DynamicShapesTokenGenError(RuntimeError):
    """Raised when multiple dynamic dims for the token_gen graph's sequence length are specified."""


def get_dynamic_shapes() -> DynamicShapes:
    """
    Computes a mapping from human-readable dimension names to the DimInfo name of a dynamic dimension used in Modelzoo
    that is registered with :class:`SambaSession`. This function requires that dynamic dims have already been registered
    to :class:`SambaSession` via :meth:`samba.session.load_dynamic_dims_config
    <sambaflow.samba.session.SambaSession.load_dynamic_dims_config>` or :meth:`samba.session.create_dynamic_dim
    <sambaflow.samba.session.SambaSession.create_dynamic_dim>`.

    Currently, only one batch_size and one sequence_length_token_gen dynamic dimension are allowed. You can specify
    multiple sequence_length_cache_gen dynamic dimensions to compile multiple cache_gen graphs with dynamic sequence
    lengths, but the max_value of the sequence_length_cache_gen dynamic dimension must be unique.

    Example:
        dynamic_shapes = DynamicShapes.create()
        dynamic_shapes.batch_size = "batch_size_min_1_max_16_factor_1"
        dynamic_shapes.sequence_length_cache_gen = {
                1024: "sequence_length_cache_gen_min_512_max_1024_factor_512",
                4096: "sequence_length_cache_gen_min_1024_max_4096_factor_1024"
            }
        dynamic_shapes.sequence_length_token_gen = "sequence_length_token_gen_min_1024_max_4096_factor_1024"

    Raises:
        DynamicShapesBatchSizeError: two dynamic dims for batch size were found
        DynamicShapesCacheGenDuplicateError: two dynamic dims for the cache_gen graph's sequence length with the same
            max size were found
        DynamicShapesTokenGenError: two dynamic dims for the token_gen graph's sequence length were found

    Returns:
        A :class:`DynamicShapes` object that maps the human-readable dimension names to the dynamic dimension name
        registered with :class:`SambaSession`.
    """
    session_dynamic_dims: Dict[str, 'samba.sambatensor.DimInfo'] = samba.session.dynamic_dims
    dynamic_shapes = DynamicShapes.create()

    for dim_name, dim_info in session_dynamic_dims.items():
        if dim_name.startswith("batch_size"):
            if dynamic_shapes.batch_size is not None:
                raise DynamicShapesBatchSizeError(
                    f"Found two batch_size dynamic dims: {dynamic_shapes.batch_size} and {dim_name}, expect only 1")
            dynamic_shapes.batch_size = dim_name
        elif dim_name.startswith("sequence_length_cache_gen"):
            if dim_info.max in dynamic_shapes.sequence_length_cache_gen:
                raise DynamicShapesCacheGenDuplicateError(
                    f"Found two sequence_length_cache_gen dims with the same max value: {dynamic_shapes.sequence_length_cache_gen[dim_info.max]} and {dim_name}"
                )
            dynamic_shapes.sequence_length_cache_gen[dim_info.max] = dim_name
        elif dim_name.startswith("sequence_length_token_gen"):
            if dynamic_shapes.sequence_length_token_gen is not None:
                raise DynamicShapesTokenGenError(
                    f"Found two sequence_length_token_gen dynamic dims: {dynamic_shapes.sequence_length_token_gen} and {dim_name}, expect only 1"
                )
            dynamic_shapes.sequence_length_token_gen = dim_name

    return dynamic_shapes
