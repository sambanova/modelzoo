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

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import os
import typing
from typing import Any, Callable, DefaultDict, List, Optional, Tuple, TypeVar

from sambanova_modelzoo.models.utils import is_jit, logger_info

__all__ = [
    "SNNotSupportedError",
    "SNPatch",
    "SNPatchMode",
    "sn_patch_inline",
    "sn_patch_not_supported",
    "sn_patch_post_process_result",
    "sn_patch_post_process_self",
    "sn_patch_pre_process_self",
    "sn_patch_replace",
    "sn_patch_replacements",
]

_ENV_PATCH_OFF_NAME = 'SN_PATCH_GLOBAL_OFF'

# When True, functions that are patched are logged at the logging level debug
_LOG_FUNCTION_PATCHING = False


def is_globally_disabled() -> bool:
    """
    Returns: True if and only if the environment variable SN_PATCH_GLOBAL_OFF is either 1 or true (case insensitive)
    """
    if not is_jit():
        override_env = os.environ.get(_ENV_PATCH_OFF_NAME, "")
    else:
        # The samba JIT mode does not work with accessing environment variables.
        override_env = ""
    return override_env == "1" or override_env.lower() == "true"


@contextlib.contextmanager
def testing_disable_all_patches():
    """
    A context manager for running code with patches disabled.
    Example:
            with testing_disable_all_patches():
                # run some model code with patches off.
    """
    old_value = os.environ.get(_ENV_PATCH_OFF_NAME)
    os.environ[_ENV_PATCH_OFF_NAME] = '1'
    try:
        yield
    finally:
        if type(old_value) is str:
            os.environ[_ENV_PATCH_OFF_NAME] = old_value
        else:
            del os.environ[_ENV_PATCH_OFF_NAME]


class SNPatchMode(enum.Enum):
    """Selection if multiple lambdas in a list of patches should be mutually exclusive or
       if the first one matching should win.
    """
    Exclusive = enum.auto()
    FirstWins = enum.auto()


def _get_argument_dictionary(signature: inspect.Signature,
                             args,
                             kwargs,
                             additional_arguments: typing.Dict[str, Any] = {}) -> DefaultDict[str, Any]:
    """Put all arguments into a dictionary, including values for the default-value arguments bound in the
    supplied signature.

    Args:
        signature: The signature of the called function
        args: The supplied args, must match args to a function matching the signature
        kwargs: The supplied kwargs, must match kwargs to a function matching the signature
        additional_arguments: Additional arguments to put into the argument dictionary

    Returns:
        Dictionary from all arguments to their values, with default value None
    """
    callargs = signature.bind(*args, **kwargs)
    callargs.apply_defaults()
    all_arguments = {
        **{k: v
           for (k, v) in callargs.arguments.items() if k != "kwargs"},
        **callargs.arguments.get("kwargs", {}),
        **additional_arguments
    }
    all_arguments_with_default = collections.defaultdict(lambda: None, all_arguments)
    return all_arguments_with_default


def _check_condition(func: Optional[Callable[..., bool]], args: DefaultDict[str, Any]) -> bool:
    """Check if this patch should be applied.

    Args:
        func: An function to use to check the condition. If no functions is given, handle as always true.
        args: A default dictionary of all the arguments (including keyword arguments).

    Returns:
        True if there is no `func` condition or if the `func` condition returns True.
    """
    if func is None:
        return True

    args, kwargs = _build_argument_list(func, args)
    checked_condition = func(**args, **kwargs)
    if not isinstance(checked_condition, bool):
        raise TypeError("The func function must return a Boolean, returned "
                        f"{checked_condition} of type {type(checked_condition)}.")
    return checked_condition


def _build_argument_list(func: Callable[..., Any],
                         args: DefaultDict[str, Any]) -> Tuple[typing.Dict[str, Any], typing.Dict[str, Any]]:
    """Build dictionary that binds arguments for calling `func` from value mappings in `args`

    Args:
        func: The callable to build the arguments for
        args: A default dictionary of all the arguments (including keyword arguments).

    Returns:
        True if there is no `func` condition or if the `func` condition returns True.
    """
    call_arguments = {}
    func_signature = inspect.signature(func)

    add_kwargs_as_argument = False
    add_kwargs_as_kwargs = False

    for name, parameter in func_signature.parameters.items():
        if name in args:
            call_arguments[name] = args[name]
        elif name == "all_args":
            call_arguments[name] = args
        elif name == "config" and hasattr(args["self"], "config"):
            call_arguments[name] = args["self"].config
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # Note, important that this case is before the below check for "kwargs" name, as
            # both kwargs and **kwargs will have the same name
            add_kwargs_as_kwargs = True
        elif name == "kwargs":
            add_kwargs_as_argument = True
        else:
            raise TypeError(f"The required argument {name} in the function signature is missing in the available args "
                            f"(available: {list(args.keys())})")

    if add_kwargs_as_argument:
        call_arguments["kwargs"] = {k: v for k, v in args.items() if k not in call_arguments}
    if add_kwargs_as_kwargs:
        kwargs = {k: v for k, v in args.items() if k not in call_arguments}
    else:
        kwargs = {}

    return call_arguments, kwargs


RetType = TypeVar("RetType")


class SNNotSupportedError(Exception):
    """Error that indicates that an operation is not supported"""


def sn_patch_not_supported(
        *,  # Require that all arguments are named
        not_supported_if: Optional[Callable[..., bool]] = None,
        description: Optional[str] = None,
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator that dynamically disables the decorated function if the `not_uspported_if` function returns true.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that
    a function is not valid to call.

    The `not_supported_if` lambda can take any argument that is supplied to the decorated function,
    but does not need to take all of them. In addition, there are two argument names that have special meaning:
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import sn_patch_not_supported
    ...
    ... @sn_patch_not_supported(not_supported_if=lambda val: not (0 <= val <= 100),
    ...                         description="Not supported when val is outside the range [0, 100]")
    ... def foo(val: int):
    ...     return f"foo: {val}"
    >>> foo(0)
    'foo: 0'
    >>> foo(101)
    models.patch_router.SNNotSupportedError: This call of foo is not supported. Not supported when val is outside the range [0, 100]

    Args:
        not_supported_if: The condition to use for checking if the decorated function should not be called
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that dynamically throws a SNNotSupportedError if the condition is met
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            if is_globally_disabled():
                not_supported = False
            elif not_supported_if is None:
                not_supported = True
            else:
                all_args = _get_argument_dictionary(decorated_signature, args, kwargs)
                checked_condition = _check_condition(not_supported_if, all_args)
                if not isinstance(checked_condition, bool):
                    raise TypeError("The not_supported_if function must return a Boolean, returned "
                                    f"{checked_condition} of type {type(checked_condition)}.")
                not_supported = checked_condition

            if not_supported:
                raise SNNotSupportedError(f"This call of {decorated_function.__name__} is not supported."
                                          f" {description}" if description else "")

            return decorated_function(*args, **kwargs)

        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


ClassType = TypeVar("ClassType")


def sn_patch_class_not_supported(
        *,  # Require that all arguments are named
        not_supported_if: Optional[Callable[[typing.Type[ClassType]], bool]] = None,
        description: Optional[str] = None,
) -> Callable[[typing.Type[ClassType]], typing.Type[ClassType]]:
    """Decorator that dynamically disables all methods in the decorated class if the `not_supported_if` function
    returns true or if no such guard is given.

    The intended use-case is to handle cases where parts of a model are not supported yet, but we want to leave
    those parts in the code to keep the changes minimal.

    The `not_supported_if` lambda takes the decorated class as an argument

    Usage example:
    >>> >>> from sambanova_modelzoo.models.patch_router import (SNNotSupportedError, sn_patch_class_not_supported)
    ...
    ...
    ... @sn_patch_class_not_supported()
    ... class Foo:
    ...     def __init__(self):
    ...         self.message = " hello"
    ...
    >>> Foo()
    models.patch_router.SNNotSupportedError

    Args:
        not_supported_if: The condition to use for checking if the decorated function should not be called
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that makes all methods in a class not supported
    """
    def decorator(decorated_class: typing.Type[ClassType]) -> typing.Type[ClassType]:
        if not isinstance(type(decorated_class), type):
            raise TypeError("The sn_patch_class_not_supported decorator can only be applied to classes, was applied "
                            f"to a variable of type {type(decorated_class)}.")

        if not_supported_if(decorated_class) if not_supported_if is not None else True:
            for name, value in vars(decorated_class).items():
                if callable(value):
                    setattr(decorated_class, name, sn_patch_not_supported(description=description)(value))
            return decorated_class
        else:
            return decorated_class

    return decorator


def _apply_patch(function: Callable[..., RetType], patch: Optional[Callable[..., RetType]], description: Optional[str],
                 args, kwargs) -> RetType:
    """If `patch` is supplied, call it, otherwise call `function`

    Args:
        function: The base function
        patch: Optional patch
        description: Optional description of the reason for the patch
        args: The arguments to call the function/patch with
        kwargs: The keyword arguments to call the function/patch with

    Returns:
        What the called function returns
    """
    if patch is not None:
        if _LOG_FUNCTION_PATCHING:
            logger_info.debug(f"Patching func "
                              f"{function.__name__} with "
                              f"{patch.__name__}."
                              f" Reason: {description}" if description else "")
        return patch(*args, **kwargs)
    else:
        return function(*args, **kwargs)


def sn_patch_pre_process_self(
        *,  # Require that all arguments are named
        modification: Callable[..., None],
        enable_if: Optional[Callable[..., bool]] = None,
        description: Optional[str] = None,
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator that runs a modifcation step on self before the decorated function is called.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that
    some change should be applied to some part of the self parameter.

    The `enable_if` and `modification` lambdas can take any argument that is supplied to the decorated function,
    but does not need to take all of them. In addition, there are three argument names that have special meaning:
      self: Must be the first argument to `modification` and to the decorated function
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Usage example:
    >>> from typing import List, Optional
    ... from sambanova_modelzoo.models.patch_router import sn_patch_pre_process_self
    ...
    ... class Foo:
    ...     def __init__(self, vals: Optional[List[int]] = None):
    ...         self.vals = vals
    ...
    ...     def ensure_vals_initialized(self):
    ...         if self.vals is None:
    ...             self.vals = []
    ...
    ...     @sn_patch_pre_process_self(enable_if=lambda self: self.vals is None, modification=ensure_vals_initialized)
    ...     def push_val(self, val: int) -> 0:
    ...         self.vals.append(val)
    ...         return self.count()
    ...
    ...     @sn_patch_pre_process_self(modification=ensure_vals_initialized)
    ...     def count(self) -> int:
    ...         return len(self.vals)
    ...
    >>> assert Foo().vals is None
    >>> assert Foo().count() == 0
    >>> assert Foo().push_val(42) == 1
    >>> assert Foo([1, 2, 3]).push_val(42) == 4

    Args:
        modification: The modifiation function to call
        enable_if: The condition to use for checking if `modification`should be applied. If none is given,
                  that corresponds to a patch that is always applied.
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that optionally and dynamically modifies the self of the method it is applied to
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        if len(decorated_signature.parameters) == 0:
            raise TypeError(f"First argument of decorated function should be named self, was empty.")
        if (decorated_first_arg := next(iter(decorated_signature.parameters))) != "self":
            raise TypeError(f"First argument of decorated function should be named self, was {decorated_first_arg}")

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            all_args = _get_argument_dictionary(decorated_signature, args, kwargs)
            if is_globally_disabled():
                enabled = False
            elif enable_if is None:
                enabled = True
            else:
                checked_condition = _check_condition(enable_if, all_args)
                if not isinstance(checked_condition, bool):
                    raise TypeError("The enable_if function must return a Boolean, returned "
                                    f"{checked_condition} of type {type(checked_condition)}.")
                enabled = checked_condition

            if enabled:
                modification_args, modification_kwargs = _build_argument_list(modification, all_args)
                if len(modification_args) == 0:
                    raise TypeError(f"First argument of modification should be named self, was empty.")
                if (first_arg := next(iter(modification_args))) != "self":
                    raise TypeError(f"First argument of modification should be named self, was {first_arg}")
                modification(**modification_args, **modification_kwargs)

            return decorated_function(*args, **kwargs)

        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


def sn_patch_post_process_self(
        *,  # Require that all arguments are named
        modification: Callable[..., None],
        enable_if: Optional[Callable[..., bool]] = None,
        description: Optional[str] = None,
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator that runs a modifcation step on self after the decorated function has been called.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that
    some change should be applied to some part of the self parameter.

    The `enable_if` and `modification` lambdas can take any argument that is supplied to the decorated function,
    but does not need to take all of them. In addition, there are three argument names that have special meaning:
      self: Must be the first argument to `modification` and to the decorated function
      result: The result of the decorated method.
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import sn_patch_post_process_self
    ...
    ... class Message:
    ...    @sn_patch_post_process_self(enable_if=lambda message: message.isupper(),
    ...                               modification=lambda self: self.make_message_pop(),
    ...                               description="Uppercase messages should always be popping")
    ...    def __init__(self, message: str):
    ...        self.message = message
    ...
    ...    def make_message_pop(self):
    ...        self.message += "!!!!"
    ...
    >>> assert Message("hello").message == "hello"
    >>> assert Message("HELLO").message == "HELLO!!!!"

    Args:
        modification: The modifiation function to call
        enable_if: The condition to use for checking if `modification`should be applied. If none is given,
                  that corresponds to a patch that is always applied.
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that optionally and dynamically modifies the self of the method it is applied to
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        if len(decorated_signature.parameters) == 0:
            raise TypeError(f"First argument of decorated function should be named self, was empty.")
        if (decorated_first_arg := next(iter(decorated_signature.parameters))) != "self":
            raise TypeError(f"First argument of modification should be named self, was {decorated_first_arg}")

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            result = decorated_function(*args, **kwargs)

            all_args = _get_argument_dictionary(decorated_signature, args, kwargs, {"result": result})
            if is_globally_disabled():
                enabled = False
            elif enable_if is None:
                enabled = True
            else:
                checked_condition = _check_condition(enable_if, all_args)
                if not isinstance(checked_condition, bool):
                    raise TypeError("The enable_if function must return a Boolean, returned "
                                    f"{checked_condition} of type {type(checked_condition)}.")
                enabled = checked_condition

            if enabled:
                modification_args, modification_kwargs = _build_argument_list(modification, all_args)
                if len(modification_args) == 0:
                    raise TypeError(f"First argument of modification should be named self, was empty.")
                if (first_arg := next(iter(modification_args))) != "self":
                    raise TypeError(f"First argument of modification should be named self, was {first_arg}")
                modification(**modification_args, **modification_kwargs)

            return result

        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


def sn_patch_post_process_result(
        *,  # Require that all arguments are named
        modification: Callable[..., RetType],
        enable_if: Optional[Callable[..., bool]] = None,
        description: Optional[str] = None,
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator that can map the result of a function to a new value.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that
    some slight change should be made to the result to the decorated functipn. Since `self` is an argument
    to methods, this can be used to check any parts of an object.

    The `enable_if` and `modification` lambdas can take any argument that is supplied to the decorated function,
    but does not need to take all of them. In addition, there are three argument names that have special meaning:
      result: The result of the decorated method, must be the first argument of `modification`.
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import sn_patch_post_process_result
    ...
    ... def add_extra_exclamation_marks(message):
    ...    return message + "!!!!"
    ...
    ... @sn_patch_post_process_result(enable_if=lambda message: message.isupper(),
    ...                         modification=lambda result: add_extra_exclamation_marks(result))
    ... def make_message_pop(message: str) -> str:
    ...     return message + "!"
    ...
    >>> assert make_message_pop("hello") == "hello!"
    >>> assert make_message_pop("HELLO") == "HELLO!!!!!"

    Args:
        modification: The modifiation function to call, taking result as the first argument
        enable_if: The condition to use for checking if `modification`should be applied. If none is given,
                  that corresponds to a patch that is always applied.
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that optionally and dynamically modifies the arguments to the function it is applied to
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            result = decorated_function(*args, **kwargs)

            all_args = _get_argument_dictionary(decorated_signature, args, kwargs, {"result": result})
            if is_globally_disabled():
                enabled = False
            elif enable_if is None:
                enabled = True
            else:
                checked_condition = _check_condition(enable_if, all_args)
                if not isinstance(checked_condition, bool):
                    raise TypeError("The enable_if function must return a Boolean, returned "
                                    f"{checked_condition} of type {type(checked_condition)}.")
                enabled = checked_condition

            if enabled:
                modification_args, modification_kwargs = _build_argument_list(modification, all_args)
                if len(modification_args) == 0:
                    raise TypeError(f"First argument of modification should be named self, was empty.")
                if (first_arg := next(iter(modification_args))) != "result":
                    raise TypeError(f"First argument of modification should be named result, was {first_arg}")
                result = modification(**modification_args, **modification_kwargs)

            return result

        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


def sn_patch_replace(
        *,  # Require that all arguments are named
        patch: Callable[..., RetType],
        enable_if: Optional[Callable[..., bool]] = None,
        description: Optional[str] = None,
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator factory for dynamically patching a function with some other function.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that a
    different version of a function should be used. Note that for methods, self is included in the arguments
    and thus member variables of the class can be used as well in the dynamic checks.

    The `enable_if` condition is used to check if the patch should be applied. The lambda can take any argument that is
    supplied to the decorated function, but does not need to take all of them.
    In addition, there are two argument names that have special meaning:
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Works the same as :func:`sn_patches` with a single argument.

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import sn_patch_replace
    >>> def bar(val: int) -> str:
    ...     return f"bar: {val}"
    ...
    ... @sn_patch_replace(
    ...     patch=bar,
    ...     enable_if=lambda val: val >= 100,
    ...     description="When the val argument to foo is large, call bar instead of foo"
    ... )
    ... def foo(val: int) -> str:
    ...     return f"foo: {val}"
    ...
    >>> assert foo(5) == "foo: 5"
    >>> assert foo(4711) == "bar: 4711"

    Args:
        patch: The replacement function to call instead
        enable_if: The condition to use for checking if `patch`should be used. If none is given, that corresponds to
                  a patch that is always applied.
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that dynamically patches the function it is applied to
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            if is_globally_disabled():
                enabled = False
            elif enable_if is None:
                enabled = True
            else:
                all_args = _get_argument_dictionary(decorated_signature, args, kwargs)
                checked_condition = _check_condition(enable_if, all_args)
                if not isinstance(checked_condition, bool):
                    raise TypeError("The enable_if function must return a Boolean, returned "
                                    f"{checked_condition} of type {type(checked_condition)}.")
                enabled = checked_condition
            return _apply_patch(decorated_function, patch if enabled else None, description, args, kwargs)

        # keep initial original function object
        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


@dataclasses.dataclass
class SNPatch:
    """Configuration for a dynamic patch of a function.

    When the patch is used, the patched function will be called with all the same arguments as the original function.

    The `enable_if` condition is used to check if the patch should be applied. The lambda can take any argument that is
    supplied to the decorated function, but does not need to take all of them.
    In addition, there are two argument names that have special meaning:
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Attributes:
        patch: The replacement function to call instead
        enable_if: The condition to use for checking if `patch`should be used. If none is given, that corresponds to
                  a patch that is always applied.
    """
    patch: Optional[Callable[..., RetType]]
    enable_if: Optional[Callable[..., bool]] = None
    description: Optional[str] = None


def _find_patch(patches: List[SNPatch], mode: SNPatchMode,
                all_args: typing.DefaultDict[str, Any]) -> Tuple[Optional[Callable[..., RetType]], Optional[str]]:
    """Given a list of patches and the arguments, find the right patch function, if any is matching.

    Args:
        patches: The list of patches to
        mode: The mode to use for selecting the patch
        all_args: All the arguments in a dictionary

    Returns:
        A patch that is to be applied, if any, or else None
    """
    if len(patches) == 0:
        return None, None

    if mode == SNPatchMode.Exclusive:
        conditions = [_check_condition(p.enable_if, all_args) for p in patches]
        enabled_patches_count = sum(conditions)
        assert enabled_patches_count <= 1, "At most one patch can be active at one time"
        if enabled_patches_count > 0:
            enabled_patch = conditions.index(True)
            return patches[enabled_patch].patch, patches[enabled_patch].description
        return None, None
    elif mode == SNPatchMode.FirstWins:
        for patch in patches:
            if _check_condition(patch.enable_if, all_args):
                return patch.patch, patch.description
        return None, None
    else:
        raise ValueError(f"Invalid mode {mode}")


def sn_patch_replacements(
        *,  # Require that all arguments are named
        mode: SNPatchMode = SNPatchMode.Exclusive,
        patches: List[SNPatch],
) -> Callable[[Callable[..., RetType]], Callable[..., RetType]]:
    """Decorator factory for dynamically patching a function with some other function.

    The intended use-case is to handle cases where some dynamic configuration parameter indicates that a
    different version of a function should be used. Note that for methods, self is included in the arguments
    and thus member variables of the class can be used as well in the dynamic checks.

    When one of the patches in the list returns true from the condition, the corresponding patch is used instead of the
    decorated function. If the mode is `Exclusive` (default) only one of the patches is allowed to be active for
    each call, so the list of patches must be mutually exclusive. If the mode is `FirstWins`, then the first patch
    (if any) that is true will be applied.

    The `enable_if` lambdas in the `patches` can take any argument that is supplied to the decorated function, but
    does not need to take all of them. In addition, there are two argument names that have special meaning:
      all_args : The full set of arguments as a defaultdict with default value None.
          Sometimes useful to forward logic to another function.
      config : If the decorated function has a self argument, and that self argument has a config member, use this.
          This is useful in many models where the class has a config variable containing relevant configuration that
          is not passed as arguments to specific methods

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import SNPatch, SNPatchMode, sn_patch_replacements
    >>> def bar(val: int) -> str:
    ...     return f"bar: {val}"
    ...
    ... def baz(val: int) -> str:
    ...     return f"baz: {val}"
    ...
    ... @sn_patch_replacements(
    ...     mode=SNPatchMode.FirstWins,
    ...     patches=[
    ...         SNPatch(
    ...             patch=bar,
    ...             enable_if=lambda val: val >= 100,
    ...             description="When the val argument to foo is large, call bar instead of foo"
    ...         ),
    ...         SNPatch(
    ...             patch=baz,
    ...             enable_if=lambda val: val >= 10,
    ...             description="When the val argument to foo is somewhat large, call baz instead of foo"
    ...         )
    ...     ]
    ... )
    ... def foo(val: int) -> str:
    ...     return f"foo: {val}"
    ...
    >>> assert foo(5) == "foo: 5"
    >>> assert foo(42) == "baz: 42"
    >>> assert foo(4711) == "bar: 4711"

    Args:
        patches (List[SNPatch]): The patches to check and potentially use
        mode: The to use when checking which patch to apply

    Returns:
        A decorator that dynamically patches the function it is applied to
    """
    def decorator(decorated_function: Callable[..., RetType]) -> Callable[..., RetType]:
        decorated_signature = inspect.signature(decorated_function)

        @functools.wraps(decorated_function)
        def wrapper(*args, **kwargs):
            all_args = _get_argument_dictionary(decorated_signature, args, kwargs)

            if is_globally_disabled():
                found_patch, description = None, None
            else:
                found_patch, description = _find_patch(patches, mode, all_args)

            return _apply_patch(decorated_function, found_patch, description, args, kwargs)

        wrapper.__original_function__ = getattr(decorated_function, '__original_function__', decorated_function)
        return wrapper

    return decorator


def sn_patch_inline(enable_if: Callable[[], bool],
                    description: Optional[str] = None) -> Callable[[Callable[[], RetType]], Callable[[], RetType]]:
    """Decorator factory for conditionally applying a function directly at the definition site.

    In general, using an if-statement represents the same functionality with less magic. However, using
    `sn_patch_inline` semantically marking the usages as different from normal if-statements, and can be easily
    searched for similar to usages of `sn_patch` and `sn_patches` sincel all three share the prefix `sn_patch`.

    Usage example:
    >>> from sambanova_modelzoo.models.patch_router import sn_patch_inline
    ...
    ... def foo(val: int) -> str:
    ...     result = f"foo: {val}"
    ...
    ...     @sn_patch_inline(
    ...         enable_if=lambda: val > 100,
    ...         description="Change the value of result when the val argument is large"
    ...     )
    ...     def bar():
    ...         nonlocal result # nonlocal enables this method to change the result variable
    ...         result = f"bar: {val}"
    ...
    ...     return result
    ...
    >>> assert foo(5) == "foo: 5"
    >>> assert foo(4711) == "bar: 4711"

    Args:
        enable_if: A callable that is used to check if the decorated function should be applied
        description: Non-functional argument used to place documentation in-place for what the patch does.

    Returns:
        A decorator that may eagerly run the function decorated based on the result of 'condition' and otherwise is
        an identity decorator
    """
    def decorator(decorated_function: Callable[[], RetType]) -> Callable[[], RetType]:
        if not is_globally_disabled() and enable_if():
            decorated_function()
        return decorated_function

    return decorator
