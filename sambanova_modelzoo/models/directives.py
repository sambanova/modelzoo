# Copyright 2023-2024 SambaNova Systems, Inc.
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

from contextlib import nullcontext
from typing import Any, ContextManager, Dict, List, Optional, Union

import torch
from sambanova_modelzoo.models.schema import HyperfunctionDirective
from sambanova_modelzoo.models.utils import is_jit


def add_directives(directive: Optional[Union[HyperfunctionDirective, Dict[str, Any]]]
                   ) -> ContextManager['hyperfunction']:
    """
    Add directive context. Do nothing for JIT frontends currently.
    Args:
        directive: Pydantic hyperfunction directive model.
    Returns:
        A context manager that add the directive to Samba.
    """
    if not is_jit():
        import sambaflow.samba as samba
        if isinstance(directive, HyperfunctionDirective):
            directive = directive.model_dump()
        return samba.session._add_directives(directive)
    return nullcontext()


def op_fusion(*args, **kwargs) -> ContextManager:
    """
    op_fusion context manager. Do nothing for JIT frontends currently.
    """
    if not is_jit():
        from sambaflow.samba.directives import op_fusion as samba_op_fusion
        return samba_op_fusion(*args, **kwargs)
    return nullcontext()


def opfusion_id(ids: Union[str, List[str]]):
    if not is_jit():
        # TODO (zijingg) Support list type for opfusion_id
        assert isinstance(ids, str), f"opfusion_id only supports string type for now, got {type(ids)}"
        import sambaflow.samba as samba
        key = 'opfusion_id'
        # TODO (zijingg) Support appending new ids on top of existing ones
        assert key not in samba.session._context_directives, f"Appending more opfusion_ids is currently not supported"
        return samba.session._add_directives({key: ids})
    return nullcontext()


def add_aux_tensor(tensor: torch.Tensor, name: str) -> None:
    """
    Add add_aux_tensor from samba. Do nothing for JIT frontends currently.
    Args:
        tensor: the corresponding auxiliary torch tensor to be added.
        name: user-specified name for the SambaTensor.
    """
    if not is_jit():
        import sambaflow.samba as samba
        return samba.session.add_aux_tensor(tensor, name)


def is_valid_aux_tensor_name(name: str) -> bool:
    """
    Add add_aux_tensor from samba. Do nothing for JIT frontends currently.
    Args:
        tensor: the corresponding auxiliary torch tensor to be added.
        name: user-specified name for the SambaTensor.
    """
    if not is_jit():
        import sambaflow.samba as samba
        return samba.session.is_valid_aux_tensor_name(name)
