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

import ast
import glob
import importlib
import pathlib
import types
from typing import Callable, Optional, Set, Tuple, Type

# TODO: Explain what this file does in a sentence or two.


def find_plugin_files(files_matcher_glob: str, base_class: Type, predicate: Callable[[str, str], bool]) -> Set[str]:
    """
    Recursively import all plugins (provided is matching glob) so that they register in the
    ConfigurationTransformerPlugins
    TODO: I don't understand "provided is matching glob" 
    Args:
        files_matcher_glob: a glob string that matches the files to look at, e.g. models/*/conf*.py or models/**/conf.py
        base_class: the base class to look for in files, plugins inherit base_class
        predicate: this is used to determine if file should be selected based on the base_class name
    Returns:
        Set[str]: the files that were imported
    """
    matched_files = glob.glob(files_matcher_glob, recursive=True)
    plugin_files = {f for f in matched_files if predicate(f, base_class.__name__)}

    return plugin_files


def find_general_files(files_matcher_glob: str) -> Set[str]:
    """
    Recursively find files that match files_matcher_glob
    Args:
        files_matcher_glob: a glob string that matches the files to look at, e.g. models/*/conf*.py or models/**/conf.py
    Returns:
        Set[str]: the files that were imported
    """
    matched_files = glob.glob(files_matcher_glob, recursive=True)
    return set(matched_files)


def find_import_files(path_matcher_glob: str, filename_matcher_glob: str) -> Set[Tuple[str, Optional[str]]]:
    """
    find files (and optionally requirements files) to import
    Args:
        path_matcher_glob:
        filename_matcher_glob:

    Returns:
        a set of Tuple[str, Optional[str]] of files to import and their corresponding requirements file to evaluate before importing (if any)
    """
    found_import_files = set()
    matched_files = glob.glob(path_matcher_glob + '/' + filename_matcher_glob, recursive=True)
    for file in matched_files:
        requirements_file = glob.glob(str(pathlib.Path(file).resolve().parent) + '/requirements_*.py')
        found_import_files.add((file, requirements_file[0] if len(requirements_file) > 0 else None))

    return found_import_files


def _check_that_files_in_package(module_files, package_dir, package_name):
    for path in module_files:
        if not path.startswith(package_dir):
            raise ValueError(f'Module file "{path}" not part of package: {package_name}')


def import_modules(module_files: Set[str], package_dir: str, package_name: str) -> bool:
    """
    Import modules corresponding to the source files in module_files, that are located in package package_name
    Args:
        module_files: a set of absolute paths to modules to import
        package_dir: the absolute directory of the module
        package_name: the name of the base package
    Returns:
        True if all went without errors, raises an exception if not.
    """
    _check_that_files_in_package(module_files, package_dir, package_name)

    for module_path in module_files:
        import_file(module_path, package_dir, package_name)

    return True


def import_modules_with_requirements(files_with_requirements: Set[Tuple[str, Optional[str]]], package_dir: str,
                                     package_name: str) -> bool:

    module_files = {f for (f, r) in files_with_requirements}
    _check_that_files_in_package(module_files, package_dir, package_name)

    for module_path, requirements_file in files_with_requirements:
        if requirements_file is None or _is_requirement_ok(requirements_file, package_dir, package_name):
            import_file(module_path, package_dir, package_name)

    return True


def import_file(module_path: str, package_dir: str, package_name: str) -> types.ModuleType:
    file_without_base_dir_and_py = module_path[len(package_dir) + 1:-3]
    import_sub_module = file_without_base_dir_and_py.replace('/', '.')
    import_full = f'{package_name}.{import_sub_module}'
    # This will load the module and execute class definitions it if it hasn't already been imported elsewhere
    return importlib.import_module(f'{import_full}')


def _is_requirement_ok(requirements_file: str, package_dir: str, package_name: str):
    module = import_file(module_path=requirements_file, package_dir=package_dir, package_name=package_name)
    if not hasattr(module, 'is_accepted'):
        raise ValueError(
            f'The requirements file {requirements_file} does not have the expected function "is_accepted() -> bool"')

    return module.is_accepted()


def is_subclass_in_file(file_path: str, base_class_name: Optional[str] = None) -> bool:
    """
    Checks if a file contains a class that is a direct subclass of base_class_name
    Args:
        file_path: the path to the file
        base_class_name: the base class to look for
        default is None, which always returns False. The empty string also results in false.
        Provide the string to search for in the file pointed to by file_path.

    Returns: True if file contains a direct subclass of base_class_name, else False
    """
    if base_class_name is None or base_class_name == "":
        return False

    with open(file_path, 'r') as file:
        source_code = file.read()
        code_tree = ast.parse(source_code)

        for node in ast.walk(code_tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == base_class_name:
                        return True
    return False
