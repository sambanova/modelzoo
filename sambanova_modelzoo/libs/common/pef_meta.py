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

from typing import Any, Dict

from pydantic import BaseModel
from sambanova_modelzoo.libs.common.samba_schema import SambaConfig

from sambaflow.samba.argconf.args import CommonArgs, RunArgs
from sambaflow.samba.utils.pef_utils import PefInfoStruct, get_pef_info

APP_ARGS_KEY = 'pydantic_app_args'


def to_pef_meta_dict(cfg: SambaConfig, recompile_args_only: bool = True) -> Dict[str, Any]:
    """
    Convert a pydantic model to a dictionary for PEF metadata.
    Optionally filter on args that make this PEF unique (require recompile).

    Args:
        cfg: A pydantic model that inherits from SambaConfig (contains samba_compile & samba_run).
        recompile_args_only: For non-samba args, whether to only include args marked requires_recompile=True.

    Returns:
        A nested dict with top-level keys being samba compile arguments (all samba args are included)
        and dict[APP_ARGS_KEY] being the app level pydantic model args
    """

    if cfg.command != "compile":
        raise RuntimeError("PEF metadata should only be constructed during compile")

    # Include samba args as-is
    metadata_dict = {**cfg.samba_compile.model_dump()}

    # Filter for recompile_args_only
    model_dump = cfg.model_dump()
    if recompile_args_only:
        model_dump = filter_for_recompile_args(model_dump, cfg)

    # Make sure not to reconsider samba args
    for field_to_remove in SambaConfig.model_fields:
        model_dump.pop(field_to_remove, None)

    # Include app args in the metadata dict
    metadata_dict[APP_ARGS_KEY] = model_dump

    return metadata_dict


def filter_for_recompile_args(model_dict: Dict[str, str], config: SambaConfig):
    """Delete fields from a pydantic model dump if not marked requires_recompile in the model config"""
    def _recursive_filter_helper(dct, cfg):
        for field_name, value in cfg:
            if BaseModel in type(value).__mro__:
                # This is a pydantic model with fields of its own
                _recursive_filter_helper(dct[field_name], value)
            else:
                # This is a field, delete from dict if not requires_recompile
                field_info = cfg.model_fields[field_name]
                field_extra_info = getattr(field_info, "json_schema_extra") or {}
                requires_recompile = field_extra_info.get("requires_recompile", False)
                if not requires_recompile:
                    dct.pop(field_name, None)

    def _remove_empty_dicts(dct):
        if isinstance(dct, dict):
            return {k: v for k, v in ((k, _remove_empty_dicts(v)) for k, v in dct.items()) if v}
        elif isinstance(dct, list) or isinstance(dct, tuple):
            return [v for v in map(_remove_empty_dicts, dct) if v]
        return dct

    _recursive_filter_helper(model_dict, config)
    return _remove_empty_dicts(model_dict)


def from_pef_meta_dict(cfg: SambaConfig) -> SambaConfig:
    """Read app args and samba run args from the PEF metadata"""
    cfg = update_app_args_from_pef(cfg)
    cfg.samba_run = update_common_samba_args_from_pef(cfg.samba_run)

    return cfg


def _update_dict(original, update):
    """
    Replacement for dict.update() for nested dicts.
    Needed because {'a': {'b': '1'}}.update({'a': {'c': '2'}}) would delete argument a.b
    """
    for key, value in update.items():
        if isinstance(value, dict):
            _update_dict(original[key], value)
        else:
            original[key] = value


def update_common_samba_args_from_pef(run_model: RunArgs) -> RunArgs:
    """
    Update the RunArgs pydantic model by reading compile args from pef metadata.
    Only updates fields found in Samba's CommonArgs model, as they are common to CompileArgs & RunArgs.
    Meant to be used during a run to avoid providing samba arguments twice.
    """

    fields_to_read_from_pef = [k for k in run_model.model_fields if k in CommonArgs.model_fields]

    args_in_pef = {}

    for field in fields_to_read_from_pef:
        # PefInfoStruct contains values for mock runtime (read from run_model)
        mock_value = PefInfoStruct(getattr(run_model, field))
        try:
            value_in_pef = get_pef_info(run_model.pef, {field: mock_value})
            args_in_pef.update(value_in_pef)
        except ValueError:
            # This value was not embedded in the PEF during compile
            pass

    args_dict = run_model.model_dump()
    _update_dict(args_dict, args_in_pef)

    # Recreate the pydantic model so that args are verified
    updated_model = RunArgs(**args_dict)

    return updated_model


def update_app_args_from_pef(cfg: SambaConfig) -> SambaConfig:
    """Update the root pydantic model for the app. Does not modify cfg during mock mode."""

    # If mock, don't update any app args
    mock_value = PefInfoStruct({})
    try:
        app_args = get_pef_info(cfg.samba_run.pef, {APP_ARGS_KEY: mock_value})
        app_args = app_args[APP_ARGS_KEY]
    except ValueError:
        app_args = {}

    if app_args:
        model_dump = cfg.model_dump()
        _update_dict(model_dump, app_args)
        cfg = type(cfg)(**model_dump)

    return cfg
