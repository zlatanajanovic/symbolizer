"""Top-level package for SYMBOLIZER.

Subpackages are imported lazily so lightweight package imports do not require
planner/search dependencies.
"""

from importlib import import_module

__all__ = ["ssr_parsing", "search_eval"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
