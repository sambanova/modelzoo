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

from contextlib import nullcontext
from typing import ContextManager, Optional

from sambanova_modelzoo.models.schema import HyperfunctionDirective
from sambanova_modelzoo.models.utils import is_jit


def add_directives(directive: Optional[HyperfunctionDirective]) -> ContextManager['hyperfunction']:
    """
    Add directive context. Do nothing for JIT frontends currently.
    Args:
        directive: Pydantic hyperfunction directive model.
    Returns:
        A context manager that add the directive to Samba.
    """
    if not is_jit():
        import sambaflow.samba as samba
        return samba.session._add_directives(directive.model_dump() if directive is not None else None)
    return nullcontext()


def op_fusion(*args, **kwargs) -> ContextManager:
    """
    op_fusion context manager. Do nothing for JIT frontends currently.
    """
    if not is_jit():
        from sambaflow.samba.directives import op_fusion as samba_op_fusion
        return samba_op_fusion(*args, **kwargs)
    return nullcontext()
