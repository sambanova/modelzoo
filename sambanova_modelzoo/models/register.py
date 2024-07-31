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

import pathlib

from sambanova_modelzoo.models.module_import_tool import find_import_files, import_modules_with_requirements
from sambanova_modelzoo.models.utils import is_jit


def conditional_register():
    """
    Go through all `model/*` folders and import the __register__.py if and only if
    the (optional) `requirements_*.py` file's function `is_accepted()` evaluates to true.

    __register__.py is place to register models for the following use cases:
    inputs setup for run: LLM model implements subclass of CachedInferenceRuntime so that
                          at minimum the input tensor name is identical with tracer.
    automodel and autoconfig: For models wish to be registered with Huggingface utils
                              for config and checkpoint loading
    All modules in __register__.py should NOT contain Sambaflow related packages so it
    can safely be imported for both Samba and JIT frontends (JIT cannot be loaded with Sambaflow)

    __register_samba__.py is used to register Samba-frontend-only modules:
    inputs setup for tracing: LLM model implements subclass of CachedInferenceTracer so that
                              the compiler has way to do tracing in a model agnostic way

    This is used for:
    1. apps that does not wish to use the latest and greatest of transformer that 
       includes every model. But instead just A particular transformer version that is
       new enough for the specific model that it cares about
    2. apps that does not even use any model at all but some library functions such as
       custom_ops e.t.c.
    """
    this_dir = str(pathlib.Path(__file__).resolve().parent)
    files = find_import_files(path_matcher_glob=this_dir + '/*', filename_matcher_glob='__register__.py')
    import_modules_with_requirements(files_with_requirements=files, package_dir=this_dir, package_name=__package__)

    if not is_jit():
        files = find_import_files(path_matcher_glob=this_dir + '/*', filename_matcher_glob='__register_samba__.py')
        import_modules_with_requirements(files_with_requirements=files, package_dir=this_dir, package_name=__package__)
