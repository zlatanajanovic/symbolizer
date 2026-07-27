# iw_probabilistic.py

import heapq
import os
import time
from collections import deque
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from .closed_list import ClosedList
from .novelty import Novelty
from .search import SearchAlgorithm


def _state_content_key(state: Any) -> Any:
    """Return a hashable content key for `state` that does NOT hold a strong reference.

    For VLM-grounded states, `_key` is a tuple of sorted
    literal strings — independent of the Python object identity.  Using this
    key in visited sets allows CPython's reference counter to GC state objects
    naturally, which triggers the weakref.finalize snapshot-delete callback.

    For PDDLGym-style namedtuple states there is no `_key`; the state itself
    is hashable by content and can be used directly.
    """
    if hasattr(state, '_key'):
        return state._key
    return state


class IWProbabilistic(SearchAlgorithm):
    """
    Iterated Width (IW) search algorithm that uses a closed list and a novelty class.
    """

    def __init__(self, 
                 max_width: int = 2, 
                 lambda_param: float = 1.0,
                 closed_list: ClosedList = None, 
                 novelty: Novelty = None,
                 max_num_observations: int = 1000
                 ):
        """
        Initialize the IWProbabilistic algorithm.

        Args:
            max_width: The maximum width to consider.
            lambda_param: The decay rate for the novelty probability (not explicitly used here, but can be integrated).
            closed_list: A ClosedList instance.
            novelty: A Novelty instance.
        """
        self.max_width = max_width
        self.lambda_param = lambda_param
        self.closed_list = closed_list
        self.novelty = novelty
        self.max_num_observations = max_num_observations

        # A dictionary for counting feature tuples if needed
        self.feature_tuple_counts: Dict[Tuple[Any, ...], int] = {}

    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        """
        Perform the IWProbabilistic to find a path to a goal state.

        Args:
            simulator: The simulator to interact with.
            initial_state: The initial state.

        Returns:
            A tuple with a list of actions representing the path to a goal state and the number of observations.
        """
        num_observations = 0
        
        for width in range(1, self.max_width + 1):
            # print(f"Starting IWProbabilistic with width={width}")
            result, num_obs = self._iw_probabilistic(simulator, initial_state, width)
            num_observations += num_obs
            if result:  # non-empty plan found
                return result, num_observations
        return [], num_observations  # No plan found

    def _iw_probabilistic(self, simulator: Any, initial_state: Any, width: int) -> Tuple[List[Any], int]:
        open_list = []
        counter = 0  # To maintain insertion order
        initial_observation = simulator.make_observation(initial_state)
        initial_atoms = initial_observation['atoms']
        initial_atom_probs = simulator.get_state_probability(initial_state)
        initial_feature_tuples = self._extract_feature_tuples(initial_atoms, width)
        observed_state = simulator.atoms_to_state(initial_observation)
        
        # Cache to store observation data for each state
        observation_cache = {}
        observation_cache[initial_state] = (observed_state, initial_atoms, initial_atom_probs, initial_feature_tuples)
        
        # Set to keep track of visited states
        visited_states = set()

        # Compute initial novelty
        initial_novelty = self.novelty.compute_novelty(
            observation=observed_state,
            feature_tuples=initial_feature_tuples,
            atom_probs=initial_atom_probs,
            closed_list=self.closed_list
        )

        heapq.heappush(open_list, (-initial_novelty, counter, (initial_state, [])))
        counter += 1
        num_observations = 1  # We've already made one observation for the initial state

        while open_list:
            # Pop the state with the highest novelty
            neg_novelty, _, (state, path) = heapq.heappop(open_list)
            
            # Skip if we've already visited this state
            if state in visited_states:
                continue
                
            # Mark this state as visited
            visited_states.add(state)
            
            # Get observation data from cache or make a new observation
            if state in observation_cache:
                observed_state, atoms, atom_probs, feature_tuples = observation_cache[state]
            else:
                observation = simulator.make_observation(state)
                num_observations += 1
                atoms = observation['atoms']
                atom_probs = simulator.get_state_probability(state)
                feature_tuples = self._extract_feature_tuples(atoms, width)
                observed_state = simulator.atoms_to_state(observation)
                observation_cache[state] = (observed_state, atoms, atom_probs, feature_tuples)
            
            # Recompute novelty using the current closed list state
            state_novelty_value = self.novelty.compute_novelty(
                observation=observed_state, 
                feature_tuples=feature_tuples,
                atom_probs=atom_probs,
                closed_list=self.closed_list
            )

            # Determine if the state is novel using the Novelty class
            is_novel = self.novelty.is_novel(state_novelty_value)
            # Update closed list with each feature tuple
            for ft in feature_tuples:
                ft_key = frozenset(ft)
                # Compute joint probability
                ft_prob = self._compute_joint_probability(ft, atom_probs)
                self.closed_list.update(
                    state=state,
                    atom_probs={ft_key: ft_prob}
                )
            if simulator.is_goal(state):
                return path, num_observations
            
            if num_observations > self.max_num_observations:
                return [], num_observations 

            if not is_novel:
                continue  # Skip expanding this state

            # Update feature tuple counts if desired
            if is_novel:
                for ft in feature_tuples:
                    ft_key = frozenset(ft)
                    self.feature_tuple_counts[ft_key] = self.feature_tuple_counts.get(ft_key, 0) + 1

            # Expand the state by applying all possible actions
            for action in simulator.get_actions(state):
                next_state = simulator.step(state, action)
                
                # Skip already visited states
                if next_state in visited_states:
                    continue
                    
                # Use current state's novelty as estimate for next state's novelty
                heapq.heappush(open_list, (-state_novelty_value, counter, (next_state, path + [action])))
                counter += 1

        return [], num_observations  # No plan found
     
    def _compute_joint_probability(self, feature_tuple: Tuple[Any, ...], atom_probs: Dict[Any, float]) -> float:
        """
        Compute the joint probability of a feature tuple based on atom probabilities.
        
        DESIGN DECISION: We use the average of atom probabilities instead of the product.
        
        Rationale for using average instead of product:
        1. Product assumption (P(A,B) = P(A)*P(B)) assumes independence between atoms,
           which is often violated in PDDL domains (e.g., on(A,B) and clear(A) are correlated).
        2. Product of many small probabilities causes numerical underflow, especially
           for higher-width feature tuples (width > 2).
        3. Average provides smoother novelty gradients during search, preventing
           states from being prematurely pruned due to one low-probability atom.
        4. Empirically, average performs better in our VLM-based observation setting
           where individual atom probabilities from the model are noisy estimates.
        
        If you need true independence-based joint probability, uncomment the product
        implementation below and handle underflow with log-sum-exp.

        Args:
            feature_tuple: A tuple of atoms forming a feature tuple.
            atom_probs: A dictionary mapping atoms to their observation probabilities.

        Returns:
            The joint probability of the feature tuple (approximated as average).
        """
        # Product-based implementation (assumes independence):
        # joint_prob = 1.0
        # for atom in feature_tuple:
        #     joint_prob *= atom_probs.get(atom, 1e-12)  # Small default to avoid zero
        # return joint_prob
        
        # Average-based implementation (more robust to noise):
        if not feature_tuple:
            return 0.0
        joint_prob = 0.0
        for atom in feature_tuple:
            atom_prob = atom_probs.get(atom, 0.0)
            if atom_prob == 0.0:
                return 0.0  # If any atom has zero probability, tuple is impossible
            joint_prob += atom_prob
        return joint_prob / len(feature_tuple)
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def _extract_feature_tuples_cached(atoms_tuple: Tuple[Any, ...], width: int) -> Tuple[Tuple[Any, ...], ...]:
        """
        Cached feature tuple extraction using list comprehensions.

        Args:
            atoms_tuple: A tuple of sorted atoms.
            width: The width parameter.

        Returns:
            A tuple of feature tuples.
        """
        return tuple(
            ft for w in range(1, width + 1)
            for ft in combinations(atoms_tuple, w)
        )

    def _extract_feature_tuples(self, atoms: List[Any], width: int) -> List[Tuple[Any, ...]]:
        """
        Optimized extraction of feature tuples with caching and list comprehensions.

        Args:
            atoms: List of atoms representing the state.
            width: The width parameter.

        Returns:
            A list of tuples, each containing a combination of atoms.
        """
        atoms_sorted = tuple(sorted(atoms))  # Sort and convert to tuple for caching
        return list(self._extract_feature_tuples_cached(atoms_sorted, width))

    
class IWK(SearchAlgorithm):
    """
    Iterated Width (IW) algorithm with a tuple count limit (K).
    Each feature tuple can be considered novel up to K times.
    """

    def __init__(self, max_width: int = 2, K: int = 3,max_num_observations: int = 1000):
        """
        Initialize the IWK algorithm.

        Args:
            max_width: The maximum width to consider.
            K: The maximum number of times a tuple is considered novel.
        """
        self.max_width = max_width
        self.K = K
        self.max_num_observations = max_num_observations

    def search(self, simulator: Any, initial_state: Any) -> List[Any]:
        """
        Perform IWK to find a path to a goal state.

        Args:
            simulator: The simulator to interact with.
            initial_state: The initial state to start from.

        Returns:
            A list of actions representing the path to a goal state.
        """
        total_observations = 0
        for width in range(1, self.max_width + 1):
            # print(f"Starting IWK with width={width} and K={self.K}")
            result, num_observations = self._iwk(simulator, initial_state, width)
            total_observations += num_observations
            if result:  # non-empty plan found
                return result, total_observations
        return [], total_observations  # No plan found

    def _iwk(self, simulator: Any, initial_state: Any, width: int) -> List[Any]:
        tuple_counts: Dict[frozenset, int] = {}
        open_list = deque()
        # 2-tuple (state, path) for eagerly-known states (initial state only).
        # 3-tuple (parent_state, action, path) for lazy children — stepped at pop time.
        open_list.append((initial_state, []))
        num_observations = 0

        # Cache to store observation data (keyed by content, not object identity)
        observation_cache = {}
        # Visited set uses content keys so state objects can be GC'd naturally,
        # which triggers weakref.finalize snapshot-delete for VLM-grounded states.
        visited_keys: set = set()

        while open_list:
            entry = open_list.popleft()

            # --- Lazy step: resolve (parent, action, path) at pop time ---
            if len(entry) == 3:
                parent_state, action, path = entry
                state = simulator.step(parent_state, action)
                # parent_state may now go out of scope; the simulator's
                # weakref.finalize callback will delete its snapshot automatically.
            else:
                state, path = entry

            state_key = _state_content_key(state)

            # Skip if we've already visited this state
            if state_key in visited_keys:
                continue

            # Mark this state as visited
            visited_keys.add(state_key)

            # Get observation data from cache or make a new observation
            if state_key in observation_cache:
                atoms, feature_tuples = observation_cache[state_key]
            else:
                atoms = simulator.make_observation(state)['atoms']
                num_observations += 1
                # Extract all possible feature tuples up to the current width
                feature_tuples = self._extract_feature_tuples(atoms, width)
                observation_cache[state_key] = (atoms, feature_tuples)

            # Check if any feature tuple is novel (count < K)
            is_novel = False
            for ft in feature_tuples:
                ft_key = frozenset(ft)  # Use frozenset for immutability
                count = tuple_counts.get(ft_key, 0)
                if count < self.K:
                    is_novel = True
                    tuple_counts[ft_key] = count + 1  # Update count
                # No else; we still need to check other tuples

            if simulator.is_goal(state):
                return path, num_observations
            if num_observations > self.max_num_observations:
                return [], num_observations  # No plan found

            if not is_novel:
                continue

            # Lazy expansion: children are stepped at pop time, not now.
            actions = simulator.get_actions(state)
            for action in actions:
                open_list.append((state, action, path + [action]))

        return [], num_observations  # No plan found

    @staticmethod
    @lru_cache(maxsize=10000)
    def _extract_feature_tuples_cached(atoms_tuple: Tuple[Any, ...], width: int) -> Tuple[Tuple[Any, ...], ...]:
        """
        Cached feature tuple extraction using list comprehensions.

        Args:
            atoms_tuple: A tuple of sorted atoms.
            width: The width parameter.

        Returns:
            A tuple of feature tuples.
        """
        return tuple(
            ft for w in range(1, width + 1)
            for ft in combinations(atoms_tuple, w)
        )

    def _extract_feature_tuples(self, atoms: List[Any], width: int) -> List[Tuple[Any, ...]]:
        """
        Optimized extraction of feature tuples with caching and list comprehensions.

        Args:
            atoms: List of atoms representing the state.
            width: The width parameter.

        Returns:
            A list of tuples, each containing a combination of atoms.
        """
        atoms_sorted = tuple(sorted(atoms))  # Sort and convert to tuple for caching
        return list(self._extract_feature_tuples_cached(atoms_sorted, width))



class IWKHeuristic(IWK):
    """
    IWK variant that uses the LLM-guided best action ordering when expanding states.
    """

    def __init__(self, max_width: int = 2, K: int = 1, max_num_observations: int = 1000):
        super().__init__(max_width=max_width, K=K, max_num_observations=max_num_observations)

    def _iwk(self, simulator: Any, initial_state: Any, width: int) -> List[Any]:
        tuple_counts: Dict[frozenset, int] = {}
        open_list = deque()
        open_list.append((initial_state, []))
        num_observations = 0
        observation_cache = {}
        visited_states = set()

        while open_list:
            state, path = open_list.popleft()

            if state in visited_states:
                continue
            visited_states.add(state)

            if state in observation_cache:
                atoms, feature_tuples = observation_cache[state]
            else:
                atoms = simulator.make_observation(state)['atoms']
                num_observations += 1
                feature_tuples = self._extract_feature_tuples(atoms, width)
                observation_cache[state] = (atoms, feature_tuples)

            is_novel = False
            for ft in feature_tuples:
                ft_key = frozenset(ft)
                count = tuple_counts.get(ft_key, 0)
                if count < self.K:
                    is_novel = True
                    tuple_counts[ft_key] = count + 1

            if simulator.is_goal(state):
                return path, num_observations

            if num_observations > self.max_num_observations:
                return [], num_observations

            if not is_novel:
                continue

            actions = simulator.get_actions(state)
            best_action = simulator.choose_best_action(state, actions) if hasattr(simulator, "choose_best_action") else None
            if best_action and best_action in actions:
                ordered_actions = [best_action] + [a for a in actions if a != best_action]
            else:
                ordered_actions = actions

            for action in ordered_actions:
                next_state = simulator.step(state, action)
                if next_state in visited_states:
                    continue
                open_list.append((next_state, path + [action]))

        return [], num_observations


class AStarLLM(SearchAlgorithm):
    """
    A* search guided by the LLM-provided heuristic and optional action ordering.
    """

    def __init__(self, max_num_expansions: int = 5000, max_state_visits: Optional[int] = None):
        self.max_num_expansions = max_num_expansions
        if max_state_visits is None:
            max_state_visits = int(os.getenv("ASTAR_STATE_VISIT_K", "1") or "1")
        self.max_state_visits = max(1, int(max_state_visits))
        self.counter = 0

    @staticmethod
    def _state_key(state: Any) -> Tuple[str, ...]:
        """Robust state key across simulator state representations.

        Supports:
        - PDDLGym State: `.literals`
        - ViPlan BW state: `.atoms`
        - API-grounded state: `._key`
        """
        if hasattr(state, "literals"):
            return tuple(sorted(str(l) for l in state.literals))
        if hasattr(state, "atoms"):
            return tuple(sorted(str(a) for a in state.atoms))
        if hasattr(state, "_key"):
            k = getattr(state, "_key")
            if isinstance(k, tuple):
                return tuple(str(x) for x in k)
        return (str(state),)

    def _get_heuristic_cost(self, simulator: Any, state: Any, cache: Dict[Tuple[str, ...], float]) -> float:
        key = self._state_key(state)
        if key in cache:
            return cache[key]

        heuristic_info = simulator.score_state_with_heuristic(state) if hasattr(simulator, "score_state_with_heuristic") else None
        score = 1
        if heuristic_info and isinstance(heuristic_info, dict):
            score = heuristic_info.get("heuristic_score", 1)
        heuristic_cost = max(0, 10 - int(score))
        cache[key] = heuristic_cost
        return heuristic_cost

    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        open_heap = []
        parent: Dict[Tuple[str, ...], Tuple[Optional[Tuple[str, ...]], Any]] = {}
        g_scores: Dict[Tuple[str, ...], float] = {}
        heuristic_cache: Dict[Tuple[str, ...], float] = {}
        # Track how often each perceived state was expanded.
        closed_visits: Dict[Tuple[str, ...], int] = {}
        # Best g-value among expanded instances of each perceived state.
        closed_best_g: Dict[Tuple[str, ...], float] = {}

        initial_key = self._state_key(initial_state)
        g_scores[initial_key] = 0.0
        h_cost = self._get_heuristic_cost(simulator, initial_state, heuristic_cache)
        heapq.heappush(open_heap, (h_cost, h_cost, self.counter, initial_state))
        parent[initial_key] = (None, None)
        self.counter += 1

        expansions = 0

        while open_heap and expansions <= self.max_num_expansions:
            _, h_val, _, state = heapq.heappop(open_heap)
            state_key = self._state_key(state)

            visits = closed_visits.get(state_key, 0)
            if visits >= self.max_state_visits:
                continue
            closed_visits[state_key] = visits + 1
            g_here = g_scores.get(state_key, float("inf"))
            closed_best_g[state_key] = min(closed_best_g.get(state_key, float("inf")), g_here)

            if simulator.is_goal(state):
                path = []
                current_key = state_key
                while parent[current_key][0] is not None:
                    prev_key, action = parent[current_key]
                    path.append(action)
                    current_key = prev_key
                path.reverse()
                return path, expansions

            expansions += 1

            actions = simulator.get_actions(state)
            best_action = simulator.choose_best_action(state, actions) if hasattr(simulator, "choose_best_action") else None
            if best_action and best_action in actions:
                ordered_actions = [best_action] + [a for a in actions if a != best_action]
            else:
                ordered_actions = actions

            for action in ordered_actions:
                next_state = simulator.step(state, action)
                next_key = self._state_key(next_state)
                tentative_g = g_scores[state_key] + 1

                if (
                    closed_visits.get(next_key, 0) >= self.max_state_visits
                    and tentative_g >= closed_best_g.get(next_key, float("inf"))
                ):
                    continue

                current_best_g = g_scores.get(next_key, float("inf"))
                if tentative_g < current_best_g:
                    g_scores[next_key] = tentative_g
                    parent[next_key] = (state_key, action)
                    h_cost_next = self._get_heuristic_cost(simulator, next_state, heuristic_cache)
                    f_score = tentative_g + h_cost_next
                    heapq.heappush(open_heap, (f_score, h_cost_next, self.counter, next_state))
                    self.counter += 1

        return [], expansions


class IWKStar(SearchAlgorithm):
    """
    IW search that prioritizes novel states using the LLM heuristic as a tie breaker.
    """

    def __init__(
        self,
        max_width: int = 2,
        K: int = 1,
        max_num_observations: int = 1000,
    ):
        self.max_width = max_width
        self.K = K
        self.max_num_observations = max_num_observations

    @staticmethod
    def _state_key(state: Any) -> Tuple[frozenset]:
        return (frozenset(state.literals),)

    def _extract_feature_tuples(self, atoms: List[Any], width: int) -> List[Tuple[Any, ...]]:
        atoms_sorted = tuple(sorted(atoms))
        return list(
            tuple_combo
            for w in range(1, width + 1)
            for tuple_combo in combinations(atoms_sorted, w)
        )

    def _get_heuristic_cost(self, simulator: Any, state: Any, cache: Dict[Tuple[frozenset], float]) -> float:
        key = self._state_key(state)
        if key in cache:
            return cache[key]
        heuristic_info = simulator.score_state_with_heuristic(state) if hasattr(simulator, "score_state_with_heuristic") else None
        score = 1
        if heuristic_info and isinstance(heuristic_info, dict):
            score = heuristic_info.get("heuristic_score", 1)
        heuristic_cost = max(0, 10 - int(score))
        cache[key] = heuristic_cost
        return heuristic_cost

    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        total_observations = 0
        for width in range(1, self.max_width + 1):
            result, num_observations = self._iwkstar(simulator, initial_state, width)
            total_observations += num_observations
            if result:
                return result, total_observations
        return [], total_observations

    def _iwkstar(self, simulator: Any, initial_state: Any, width: int) -> Tuple[List[Any], int]:
        tuple_counts: Dict[frozenset, int] = {}
        observation_cache: Dict[Tuple[frozenset], Tuple[List[Any], List[Tuple[Any, ...]]]] = {}
        heuristic_cache: Dict[Tuple[frozenset], float] = {}
        visited_states = set()
        open_heap = []
        counter = 0
        num_observations = 0

        initial_observation = simulator.make_observation(initial_state)
        num_observations += 1
        atoms = list(initial_observation['atoms'])
        feature_tuples = self._extract_feature_tuples(atoms, width)
        observation_cache[self._state_key(initial_state)] = (atoms, feature_tuples)

        is_novel_initial = any(tuple_counts.get(frozenset(ft), 0) < self.K for ft in feature_tuples) or not feature_tuples
        h_cost = self._get_heuristic_cost(simulator, initial_state, heuristic_cache)
        heapq.heappush(open_heap, (0 if is_novel_initial else 1, h_cost, counter, initial_state, [], feature_tuples))
        counter += 1

        while open_heap:
            novelty_flag, h_val, _, state, path, feature_tuples_state = heapq.heappop(open_heap)
            state_key = self._state_key(state)

            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            if simulator.is_goal(state):
                return path, num_observations

            if novelty_flag != 0:
                continue  # Skip non-novel states

            for ft in feature_tuples_state:
                ft_key = frozenset(ft)
                tuple_counts[ft_key] = tuple_counts.get(ft_key, 0) + 1

            if num_observations > self.max_num_observations:
                return [], num_observations

            actions = simulator.get_actions(state)
            best_action = simulator.choose_best_action(state, actions) if hasattr(simulator, "choose_best_action") else None
            if best_action and best_action in actions:
                ordered_actions = [best_action] + [a for a in actions if a != best_action]
            else:
                ordered_actions = actions

            for action in ordered_actions:
                next_state = simulator.step(state, action)
                next_key = self._state_key(next_state)
                if next_key in visited_states:
                    # next_state goes out of scope here; weakref.finalize deletes its snapshot.
                    continue

                if next_key in observation_cache:
                    atoms_next, feature_tuples_next = observation_cache[next_key]
                else:
                    observation_next = simulator.make_observation(next_state)
                    num_observations += 1
                    atoms_next = list(observation_next['atoms'])
                    feature_tuples_next = self._extract_feature_tuples(atoms_next, width)
                    observation_cache[next_key] = (atoms_next, feature_tuples_next)

                is_novel = any(tuple_counts.get(frozenset(ft), 0) < self.K for ft in feature_tuples_next) or not feature_tuples_next
                if not is_novel:
                    # next_state goes out of scope; weakref.finalize deletes its snapshot.
                    continue
                h_cost_next = self._get_heuristic_cost(simulator, next_state, heuristic_cache)
                heapq.heappush(
                    open_heap,
                    (0, h_cost_next, counter, next_state, path + [action], feature_tuples_next)
                )
                counter += 1

            # state goes out of scope after this iteration; weakref.finalize
            # will delete its snapshot automatically (if the simulator supports it).

        return [], num_observations


class BFWS(SearchAlgorithm):
    """
    Best-First Width Search (BFWS).

    Combines novelty-based pruning (IW) with a symbolic goal-count heuristic:

        h(s) = number of goal atoms not yet satisfied in s

    Novel states are expanded best-first by h(s), giving goal-directed width
    search without any LLM calls.  Iterates over width 1..max_width (like IWK)
    until a solution is found.

    This implements BFWS(#g, width) as described in:
        Lipovetzky & Geffner, "Best-First Width Search", AAAI 2017.

    Works with both pddlgym-style states (state.literals / simulator.chosen_goal)
    and ViPlan-style states (state.atoms / state.goal).
    """

    def __init__(self, max_width: int = 2, K: int = 3,
                 max_num_observations: int = 1000):
        self.max_width = max_width
        self.K = K
        self.max_num_observations = max_num_observations

    # ------------------------------------------------------------------
    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        goal_atoms = self._get_goal_atoms(simulator, initial_state)
        total_obs = 0
        for width in range(1, self.max_width + 1):
            plan, obs = self._bfws(simulator, initial_state, width, goal_atoms)
            total_obs += obs
            if plan is not None:
                return plan, total_obs
        return [], total_obs

    def _bfws(
        self,
        simulator: Any,
        initial_state: Any,
        width: int,
        goal_atoms: List[str],
    ) -> Tuple[Optional[List[Any]], int]:
        """BFWS at a fixed novelty width.  Returns (plan, num_obs)."""
        tuple_counts: Dict[frozenset, int] = {}
        obs_cache: Dict[Any, Tuple[List[Any], List[Tuple[Any, ...]]]] = {}
        visited: set = set()
        open_heap: list = []
        counter = 0
        num_obs = 0

        def _enqueue(state: Any, path: List[Any]) -> None:
            """Observe *state* (once), compute h, push onto heap."""
            nonlocal num_obs, counter
            if state in visited:
                return
            if state not in obs_cache:
                if num_obs >= self.max_num_observations:
                    return
                atoms = simulator.make_observation(state)["atoms"]
                num_obs += 1
                fts = IWK._extract_feature_tuples_cached(
                    tuple(sorted(str(a) for a in atoms)), width
                )
                obs_cache[state] = (atoms, fts)
            atoms, fts = obs_cache[state]
            h = self._h(atoms, goal_atoms)
            heapq.heappush(open_heap, (h, counter, state, path, fts))
            counter += 1

        _enqueue(initial_state, [])

        while open_heap:
            h_val, _, state, path, fts = heapq.heappop(open_heap)

            if state in visited:
                continue
            visited.add(state)

            # Novelty check at pop time (lazy — more accurate than push-time).
            is_novel = any(
                tuple_counts.get(frozenset(ft), 0) < self.K for ft in fts
            ) or not fts
            if not is_novel:
                continue

            for ft in fts:
                key = frozenset(ft)
                tuple_counts[key] = tuple_counts.get(key, 0) + 1

            if simulator.is_goal(state):
                return path, num_obs

            if num_obs >= self.max_num_observations:
                return None, num_obs

            for action in simulator.get_actions(state):
                next_state = simulator.step(state, action)
                if next_state not in visited:
                    _enqueue(next_state, path + [action])

        return None, num_obs

    # ------------------------------------------------------------------
    @staticmethod
    def _h(atoms: List[Any], goal_atoms: List[str]) -> int:
        """Goal-count heuristic: number of goal atoms not yet in *atoms*."""
        if not goal_atoms:
            return 0
        atom_strs = {str(a) for a in atoms}
        return sum(1 for g in goal_atoms if str(g) not in atom_strs)

    @staticmethod
    def _get_goal_atoms(simulator: Any, initial_state: Any) -> List[str]:
        """Extract goal atoms as strings from simulator or initial state."""
        # pddlgym-style
        goal = getattr(simulator, "chosen_goal", None)
        if goal is not None:
            if hasattr(goal, "literals"):
                return [str(lit) for lit in goal.literals]
            return [str(goal)]
        # ViPlan-style: state.goal is a frozenset/set of strings
        goal = getattr(initial_state, "goal", None)
        if goal is None:
            return []
        if hasattr(goal, "literals"):
            return [str(lit) for lit in goal.literals]
        try:
            return [str(g) for g in goal]
        except TypeError:
            return [str(goal)]


class SIW(SearchAlgorithm):
    """
    Serialized Iterated Width (SIW).

    Decomposes the conjunctive goal into individual goal atoms and
    solves them one at a time using IW(k).  For each sub-goal the
    novelty table is reset, but the plan and current state carry over
    from the previous sub-goal step.

    When achieving sub-goal g_i, the termination condition is:
        g_i ∈ state.literals  AND  all previously achieved g_j still hold

    If IW(1) fails for a sub-goal, IW(2) is tried.  If all widths
    fail the algorithm gives up and returns the empty plan.
    """

    def __init__(self, max_width: int = 2, K: int = 3,
                 max_num_observations: int = 1000):
        self.max_width = max_width
        self.K = K
        self.max_num_observations = max_num_observations

    # ------------------------------------------------------------------
    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        goal_atoms = self._get_goal_atoms(simulator, initial_state)

        if not goal_atoms:
            # No decomposition available — fall back to plain IWK
            return IWK(
                max_width=self.max_width,
                K=self.K,
                max_num_observations=self.max_num_observations,
            ).search(simulator, initial_state)

        goal_atoms = self._order_goal_atoms(goal_atoms)

        plan: List[Any] = []
        current_state = initial_state
        total_obs = 0
        achieved: List[Any] = []          # sub-goals satisfied so far
        remaining = list(goal_atoms)

        while remaining:
            if total_obs >= self.max_num_observations:
                return [], total_obs

            # Skip goals already satisfied in the current state
            still_remaining = []
            for g in remaining:
                if self._literal_holds(current_state, g):
                    achieved.append(g)
                else:
                    still_remaining.append(g)
            remaining = still_remaining
            if not remaining:
                break

            next_goal = remaining[0]
            # Try with must_hold first; if that fails, relax the constraint
            # (handles Sussman's anomaly in blocksworld).
            sub_plan, obs, reached = self._iw_for_subgoal(
                simulator, current_state, next_goal, achieved,
            )
            total_obs += obs

            if not reached and achieved:
                # Retry without must_hold (relaxed SIW)
                sub_plan, obs2, reached = self._iw_for_subgoal(
                    simulator, current_state, next_goal, [],
                )
                total_obs += obs2

            if reached:
                plan.extend(sub_plan)
                for action in sub_plan:
                    current_state = simulator.step(current_state, action)
                achieved.append(next_goal)
                remaining.pop(0)
                # After relaxed step some previously achieved goals may have been
                # violated — move them back to remaining so they get re-achieved.
                still_achieved = []
                for g in achieved[:-1]:   # exclude the one we just achieved
                    if self._literal_holds(current_state, g):
                        still_achieved.append(g)
                    else:
                        remaining.append(g)  # needs to be re-achieved
                achieved = still_achieved + [next_goal]
            else:
                # Sub-goal unreachable — give up
                return [], total_obs

        if simulator.is_goal(current_state):
            return plan, total_obs
        return [], total_obs

    # ------------------------------------------------------------------
    def _iw_for_subgoal(
        self,
        simulator: Any,
        start_state: Any,
        target: Any,
        must_hold: List[Any],
    ) -> Tuple[List[Any], int, bool]:
        """Run IW(1..max_width) toward *target* while keeping *must_hold* satisfied."""
        for width in range(1, self.max_width + 1):
            sub_plan, obs = self._iw_single(
                simulator, start_state, target, must_hold, width
            )
            if sub_plan is not None:
                return sub_plan, obs, True
        return [], obs, False  # type: ignore[possibly-undefined]

    def _iw_single(
        self,
        simulator: Any,
        start_state: Any,
        target: Any,
        must_hold: List[Any],
        width: int,
    ) -> Tuple[Optional[List[Any]], int]:
        """IW(width) for a single sub-goal. Returns (plan, obs) — plan is None on failure."""
        tuple_counts: Dict[frozenset, int] = {}
        open_list: deque = deque()
        # 2-tuple (state, path) for the start state; 3-tuple (parent, action, path) for lazy children.
        open_list.append((start_state, []))
        visited_keys: set = set()
        obs_cache: Dict[Any, Tuple[set, list]] = {}
        num_obs = 0

        while open_list:
            if num_obs >= self.max_num_observations:
                return None, num_obs

            entry = open_list.popleft()

            # --- Lazy step: resolve (parent, action, path) at pop time ---
            if len(entry) == 3:
                parent_state, action, path = entry
                state = simulator.step(parent_state, action)
                # parent_state may go out of scope; weakref.finalize handles snapshot cleanup.
            else:
                state, path = entry

            state_key = _state_content_key(state)

            if state_key in visited_keys:
                continue
            visited_keys.add(state_key)

            # --- sub-goal check ---
            if self._literal_holds(state, target) and all(
                self._literal_holds(state, g) for g in must_hold
            ):
                return path, num_obs

            # --- observation ---
            if state_key in obs_cache:
                atoms, feature_tuples = obs_cache[state_key]
            else:
                atoms = simulator.make_observation(state)["atoms"]
                num_obs += 1
                feature_tuples = IWK._extract_feature_tuples_cached(
                    tuple(sorted(str(a) for a in atoms)), width
                )
                obs_cache[state_key] = (atoms, feature_tuples)

            # --- novelty check ---
            is_novel = False
            for ft in feature_tuples:
                key = frozenset(ft)
                cnt = tuple_counts.get(key, 0)
                if cnt < self.K:
                    is_novel = True
                    tuple_counts[key] = cnt + 1
            if not is_novel:
                continue

            # --- lazy expansion ---
            actions = simulator.get_actions(state)
            for action in actions:
                open_list.append((state, action, path + [action]))

        return None, num_obs

    # ------------------------------------------------------------------
    @staticmethod
    def _order_goal_atoms(goal_atoms: List[Any]) -> List[Any]:
        """Order goal atoms so that 'on(x,y)' goals come after 'incolumn(y,...)' goals.

        For blocksworld-style domains:
        - 'incolumn' atoms are positional — achieve them first
        - 'on(x,y)' atoms depend on y being positioned — put after incolumn
        - 'clear' atoms are side-effects — put last

        For other domains, returns atoms in original order.
        """
        if not goal_atoms or not isinstance(goal_atoms[0], str):
            return goal_atoms

        import re
        incolumn, on_atoms, clear_atoms, other = [], [], [], []
        for g in goal_atoms:
            name = g.split("(")[0].lower()
            if name == "incolumn":
                incolumn.append(g)
            elif name == "on":
                on_atoms.append(g)
            elif name == "clear":
                clear_atoms.append(g)
            else:
                other.append(g)

        if not incolumn and not on_atoms:
            return goal_atoms  # not a blocksworld-style domain

        # For on(x,y): put after incolumn(y,...) if such a goal exists
        # Build a dependency: on(x,y) depends on incolumn(y,...)
        incolumn_blocks = set()
        for g in incolumn:
            m = re.match(r'incolumn\((\w+),', g)
            if m:
                incolumn_blocks.add(m.group(1))

        on_early, on_late = [], []
        for g in on_atoms:
            m = re.match(r'on\(\w+,(\w+)\)', g)
            if m and m.group(1) in incolumn_blocks:
                on_late.append(g)   # depends on incolumn of target
            else:
                on_early.append(g)

        return other + incolumn + on_early + on_late + clear_atoms

    # ------------------------------------------------------------------
    @staticmethod
    def _get_goal_atoms(simulator: Any, initial_state: Any) -> List[Any]:
        """Extract individual goal literals from the simulator."""
        # pddlgym-style: chosen_goal may be a conjunction with .literals
        goal = getattr(simulator, "chosen_goal", None)
        if goal is not None:
            if hasattr(goal, "literals"):
                return list(goal.literals)
            return [goal]

        # ViPlan-style: state.goal is a frozenset/set of strings
        goal = getattr(initial_state, "goal", None)
        if goal is None:
            return []
        if hasattr(goal, "literals"):
            return list(goal.literals)
        # frozenset/set/list of strings
        try:
            return list(goal)
        except TypeError:
            return [goal]

    @staticmethod
    def _literal_holds(state: Any, literal: Any) -> bool:
        """Check whether *literal* is satisfied in *state*."""
        # pddlgym State has .literals (frozenset of Literal objects)
        lits = getattr(state, "literals", None)
        if lits is not None:
            return literal in lits

        # ViPlan-style: state.atoms is a list/set of strings
        atoms = getattr(state, "atoms", None)
        if atoms is not None:
            return literal in atoms

        # Last resort: treat state itself as a collection
        try:
            return literal in state
        except TypeError:
            return False


class BFSIW(SearchAlgorithm):
    """
    Best-First Serialized Iterated Width (BFSIW).

    Combines SIW's goal decomposition with BFWS's best-first novelty search:

    - Goals are decomposed into ordered sub-goals (like SIW).
    - Each sub-goal is solved by running IW(1..max_width) in *best-first* order,
      guided by a cheap symbolic heuristic:

          h(s) = (1 if target not in s) + (# must_hold atoms not in s)

      meaning states that are both novel and closer to the current sub-goal
      are expanded first.
    - On Sussman-anomaly cases (target unreachable with must_hold), the search
      retries without the must_hold constraint and re-queues any violated atoms.

    This eliminates the two failure modes of its parents:
    - IWK fails medium/hard because width-2 is exhausted on the full goal.
    - BFWS fails some hard problems because h only counts unsatisfied goal atoms
      without decomposing dependencies — also exhausting width-2.
    - BFSIW combines sub-goal decomposition (SIW) with directed expansion (BFWS)
      so each per-sub-goal IW call is small and goal-directed.
    """

    def __init__(self, max_width: int = 2, K: int = 3,
                 max_num_observations: int = 1000):
        self.max_width = max_width
        self.K = K
        self.max_num_observations = max_num_observations

    # ------------------------------------------------------------------
    def search(self, simulator: Any, initial_state: Any) -> Tuple[List[Any], int]:
        goal_atoms = SIW._get_goal_atoms(simulator, initial_state)

        if not goal_atoms:
            # No decomposition — fall back to plain BFWS
            return BFWS(
                max_width=self.max_width,
                K=self.K,
                max_num_observations=self.max_num_observations,
            ).search(simulator, initial_state)

        goal_atoms = SIW._order_goal_atoms(goal_atoms)

        plan: List[Any] = []
        current_state = initial_state
        total_obs = 0
        achieved: List[Any] = []
        remaining = list(goal_atoms)

        while remaining:
            if total_obs >= self.max_num_observations:
                return [], total_obs

            # Skip sub-goals already satisfied now
            still_remaining = []
            for g in remaining:
                if SIW._literal_holds(current_state, g):
                    achieved.append(g)
                else:
                    still_remaining.append(g)
            remaining = still_remaining
            if not remaining:
                break

            next_goal = remaining[0]

            # Try best-first sub-goal search with must_hold constraint
            sub_plan, obs, reached = self._bfws_for_subgoal(
                simulator, current_state, next_goal, achieved
            )
            total_obs += obs

            if not reached and achieved:
                # Retry without must_hold (Sussman's anomaly)
                sub_plan, obs2, reached = self._bfws_for_subgoal(
                    simulator, current_state, next_goal, []
                )
                total_obs += obs2

            if reached:
                plan.extend(sub_plan)
                for action in sub_plan:
                    current_state = simulator.step(current_state, action)
                achieved.append(next_goal)
                remaining.pop(0)
                # Re-queue any previously achieved atoms that were violated
                still_achieved = []
                for g in achieved[:-1]:
                    if SIW._literal_holds(current_state, g):
                        still_achieved.append(g)
                    else:
                        remaining.append(g)
                achieved = still_achieved + [next_goal]
            else:
                return [], total_obs

        if simulator.is_goal(current_state):
            return plan, total_obs
        return [], total_obs

    # ------------------------------------------------------------------
    def _bfws_for_subgoal(
        self,
        simulator: Any,
        start_state: Any,
        target: Any,
        must_hold: List[Any],
    ) -> Tuple[List[Any], int, bool]:
        """BFWS(1..max_width) toward *target* keeping *must_hold* atoms.  Returns (plan, obs, reached)."""
        obs_total = 0
        for width in range(1, self.max_width + 1):
            plan, obs = self._bfws_single(
                simulator, start_state, target, must_hold, width
            )
            obs_total += obs
            if plan is not None:
                return plan, obs_total, True
        return [], obs_total, False

    def _bfws_single(
        self,
        simulator: Any,
        start_state: Any,
        target: Any,
        must_hold: List[Any],
        width: int,
    ) -> Tuple[Optional[List[Any]], int]:
        """Best-first IW(width) for a single sub-goal.  Returns (plan, obs) — plan=None on failure.

        VLM-efficient: observations are made lazily at *pop* time, not at enqueue time.
        This matches IWK's approach — only expanded (novel) states cost a VLM call.
        Children are enqueued with their parent's h as an approximate priority so the
        heap remains ordered without paying for a VLM call per child.
        """
        tuple_counts: Dict[frozenset, int] = {}
        obs_cache: Dict[Any, Tuple[List[Any], list]] = {}
        visited: set = set()
        # heap entries: (h_approx, counter, state, path)
        open_heap: list = []
        counter = 0
        num_obs = 0

        # Enqueue start state with h=0 (unknown); it will be observed at pop time.
        heapq.heappush(open_heap, (0, counter, start_state, []))
        counter += 1

        while open_heap:
            h_approx, _, state, path = heapq.heappop(open_heap)

            if state in visited:
                continue
            visited.add(state)

            # --- Lazy observation at pop time ---
            if state not in obs_cache:
                if num_obs >= self.max_num_observations:
                    return None, num_obs
                atoms = simulator.make_observation(state)["atoms"]
                num_obs += 1
                fts = IWK._extract_feature_tuples_cached(
                    tuple(sorted(str(a) for a in atoms)), width
                )
                obs_cache[state] = (atoms, fts)
            atoms, fts = obs_cache[state]

            # --- Novelty check ---
            is_novel = any(
                tuple_counts.get(frozenset(ft), 0) < self.K for ft in fts
            ) or not fts
            if not is_novel:
                continue

            for ft in fts:
                tuple_counts[frozenset(ft)] = tuple_counts.get(frozenset(ft), 0) + 1

            # --- Sub-goal check ---
            if SIW._literal_holds(state, target) and all(
                SIW._literal_holds(state, g) for g in must_hold
            ):
                return path, num_obs

            if num_obs >= self.max_num_observations:
                return None, num_obs

            # Compute h from this state's (already observed) atoms to guide children.
            h_self = self._h_subgoal(atoms, target, must_hold)

            for action in simulator.get_actions(state):
                next_state = simulator.step(state, action)
                if next_state not in visited:
                    # Enqueue children with parent's h as approximate priority.
                    # Real h computed at pop time — no VLM call here.
                    heapq.heappush(open_heap, (h_self, counter, next_state, path + [action]))
                    counter += 1

        return None, num_obs

    # ------------------------------------------------------------------
    @staticmethod
    def _h_subgoal(atoms: List[Any], target: Any, must_hold: List[Any]) -> int:
        """Heuristic for a sub-goal step: counts unsatisfied {target} ∪ must_hold atoms."""
        atom_strs = {str(a) for a in atoms}
        h = 0
        if str(target) not in atom_strs:
            h += 1
        for g in must_hold:
            if str(g) not in atom_strs:
                h += 1
        return h


### Artefact, tb removed

# if __name__ == "__main__":

#         import sys
#         import os 
#         sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#         #add search_eval to the path
#         sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#         sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
#         sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../simulator')))
#         print(sys.path)
#         from ..simulator.pddlgym_simulator import PDDLGymSimulator
#         domain_name = "Hanoi"
#         # domain_name = "Blocks_operator_actions"
#         env_name = f"PDDLEnv{domain_name}-v0"
        
#         simulator = PDDLGymSimulator(
#         env_name=env_name,
#         problem_index=1,
#         false_negative_rate=0,
#         false_positive_rate=0,
#         chosen_goal="on(d3:default,d4:default)"
#         )
#         iwk = IWK(
#                 max_width=1,
#                 K=1,
#                 max_num_observations=100
#             )
#         initial_state = simulator.reset()
#         plan, observations = iwk.search(simulator, initial_state)
#         print(plan) 
#         print(observations)
