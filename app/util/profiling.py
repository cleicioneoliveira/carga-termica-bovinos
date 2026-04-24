# app/util/profiling.py
from __future__ import annotations

import cProfile
import io
import logging
import pstats
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

# controla se já existe um profiler ativo neste contexto
_PROFILING_ACTIVE: ContextVar[bool] = ContextVar("_PROFILING_ACTIVE", default=False)


def run_with_profile(
    func: Callable[P, R],
    *args: P.args,
    enabled: bool = True,
    sort_by: str = "cumulative",
    lines: int = 30,
    profile_file: str | Path | None = "profile.prof",
    print_stats: bool = True,
    logger_instance: logging.Logger | None = None,
    allow_nested: bool = False,
    **kwargs: P.kwargs,
) -> R:
    """
    Executa uma função com cProfile.

    Se já houver profiling ativo e allow_nested=False, executa a função
    normalmente para evitar o erro:
        ValueError: Another profiling tool is already active
    """
    if not enabled:
        return func(*args, **kwargs)

    already_active = _PROFILING_ACTIVE.get()

    if already_active and not allow_nested:
        return func(*args, **kwargs)

    profiler = cProfile.Profile()
    token = _PROFILING_ACTIVE.set(True)

    try:
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        if profile_file is not None:
            profile_path = Path(profile_file)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(profile_path))

        if print_stats:
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats(sort_by).print_stats(lines)
            output = stream.getvalue()

            if logger_instance is not None:
                logger_instance.info("Resumo do profiling:\n%s", output)
            else:
                print(output)

        return result

    finally:
        _PROFILING_ACTIVE.reset(token)


def profiled(
    *,
    enabled: bool = True,
    sort_by: str = "cumulative",
    lines: int = 20,
    profile_file: str | Path | None = None,
    print_stats: bool = True,
    logger_instance: logging.Logger | None = None,
    allow_nested: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator para aplicar cProfile em uma função.

    Se a função for chamada dentro de outra já perfilada, por padrão
    não abre um novo profiler.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return run_with_profile(
                func,
                *args,
                enabled=enabled,
                sort_by=sort_by,
                lines=lines,
                profile_file=profile_file,
                print_stats=print_stats,
                logger_instance=logger_instance,
                allow_nested=allow_nested,
                **kwargs,
            )
        return wrapper
    return decorator


def print_profile_file(
    profile_file: str | Path,
    *,
    sort_by: str = "cumulative",
    lines: int = 30,
) -> None:
    stats = pstats.Stats(str(profile_file))
    stats.strip_dirs().sort_stats(sort_by).print_stats(lines)


def profile_to_text(
    profile_file: str | Path,
    *,
    sort_by: str = "cumulative",
    lines: int = 30,
) -> str:
    stream = io.StringIO()
    stats = pstats.Stats(str(profile_file), stream=stream)
    stats.strip_dirs().sort_stats(sort_by).print_stats(lines)
    return stream.getvalue()
