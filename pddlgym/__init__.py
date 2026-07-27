# Top-level pddlgym namespace — redirect to inner package
# When running from repo root, this __init__.py is found first.
# Redirect __path__ so that 'from pddlgym.parser import ...' etc.
# resolve to the inner pddlgym/pddlgym/ package.
import os as _os
_inner = _os.path.join(_os.path.dirname(__file__), "pddlgym")
if _os.path.isdir(_inner):
    __path__ = [_inner]
    # Execute the inner package's __init__.py to register environments, etc.
    _inner_init = _os.path.join(_inner, "__init__.py")
    if _os.path.isfile(_inner_init):
        _g = dict(globals())
        _g["__file__"] = _inner_init
        exec(compile(open(_inner_init).read(), _inner_init, "exec"), _g)
        globals().update({k: v for k, v in _g.items() if not k.startswith("_")})
del _inner, _os
