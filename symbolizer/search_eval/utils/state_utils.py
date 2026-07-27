# Utils/state_utils.py

import json
from typing import Any, Dict


def state_to_json(state: Any) -> Dict:
    """
    Convert the state to JSON serializable format.

    Args:
        state: The state to convert.

    Returns:
        A JSON serializable dict representing the state.
    """
    return {"state": str(state)}


def pddl_state_to_json(state: Any) -> str:
    """Convert a PDDLGym-like state to SSR-style grounded predicate JSON."""
    grounded_predicates = []
    for atom in state.literals:
        predicate_name = atom.predicate.name
        pred = {"predicate_type": predicate_name.capitalize()}
        for i, obj in enumerate(atom.variables):
            arg_key = chr(120 + i)  # x, y, z, ...
            pred[arg_key] = {"name": obj.name}
        grounded_predicates.append(pred)
    return json.dumps({"grounded_predicates": grounded_predicates})
