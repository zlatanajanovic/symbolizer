"""Simulator package exports.

Keep optional dependencies lazy so importing concrete simulators
does not require `pddlgym` unless needed.
"""

from importlib.util import find_spec

from .simulator import *

if find_spec("pddlgym") is not None:
    from .pddlgym_simulator import *

if find_spec("pddlgym") is not None:
    from .pddlgym_openai_simulator import *

if find_spec("pddlgym") is not None:
    from .pddl_file_simulator import *

if (
    find_spec("pddlgym") is not None
    and find_spec("matplotlib") is not None
    and find_spec("imageio") is not None
):
    from .vlm_pddl_simulator import *
