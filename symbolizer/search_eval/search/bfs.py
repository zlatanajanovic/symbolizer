# Search/bfs.py

from collections import deque
from typing import Any, List, Tuple

from .search import SearchAlgorithm


def _state_content_key(state: Any) -> Any:
    """Hashable content key that does NOT hold a strong ref to the state object.

    For VLM-grounded states, `_key` is a tuple of sorted literal
    strings.  Using it in visited sets lets CPython GC the state object, which
    triggers the weakref.finalize snapshot-delete callback registered by the
    simulator.  For PDDLGym namedtuple states there is no `_key`; the state
    itself is hashable by content.
    """
    if hasattr(state, '_key'):
        return state._key
    return state

class BreadthFirstSearch(SearchAlgorithm):
    """
    Vanilla Breadth-First Search that expands the exact (fully-observable) states
    returned by the simulator.  It is intended only as a speed / node-count
    baseline – no heuristics, no novelty pruning.
    """

    def __init__(self, max_num_expansions: int = 100_000) -> None:
        """
        Args
        ----
        max_num_expansions : int
            Hard ceiling on state expansions to avoid infinite loops
            in cyclic domains.
        """
        self.max_num_expansions = max_num_expansions

    # --------------------------------------------------------------------- #
    # SearchAlgorithm API                                                   #
    # --------------------------------------------------------------------- #

    def search(
        self,
        simulator: Any,
        initial_state: Any
    ) -> Tuple[List[Any], int]:
        """
        Returns
        -------
        plan            : list[action]   – sequence leading to a goal
        num_expansions  : int            – how many states were expanded
        """
        frontier: deque[Tuple[Any, List[Any]]] = deque()
        visited_keys: set = set()

        frontier.append((initial_state, []))
        num_expansions = 0

        while frontier:
            state, path = frontier.popleft()

            # goal test ----------------------------------------------------
            if simulator.is_goal(state):
                return path, num_expansions

            # duplicate detection -----------------------------------------
            state_key = _state_content_key(state)
            if state_key in visited_keys:
                continue
            visited_keys.add(state_key)

            # expansion limit ---------------------------------------------
            num_expansions += 1
            if num_expansions >= self.max_num_expansions:
                print(f"[BFS] Reached expansion cap of {self.max_num_expansions}")
                return [], num_expansions

            # generate successors -----------------------------------------
            for action in simulator.get_actions(state):
                next_state = simulator.step(state, action)
                if _state_content_key(next_state) not in visited_keys:
                    frontier.append((next_state, path + [action]))

        # no plan ----------------------------------------------------------
        return [], num_expansions


# Backward-compatible alias
BFS = BreadthFirstSearch