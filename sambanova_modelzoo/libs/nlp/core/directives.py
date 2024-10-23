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
from typing import Any, Dict

from sambanova_modelzoo.models.utils import is_jit


def sdpa_directives(directives: Dict[str, Any]):
    """SDPA directive context"""
    if not is_jit():
        import sambaflow.samba as samba
        return samba.directives.sdpa_directives(directives)
    return nullcontext()


def disable_graphamp():
    """'Disable graphmap' directive context"""
    if not is_jit():
        import sambaflow.samba as samba
        return samba.session.disable_graphamp()
    return nullcontext()
