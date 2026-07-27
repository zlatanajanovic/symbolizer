"""Pure-symbolic PDDL simulator using PDDLEnv with custom domain/problem files."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pddlgym.core import PDDLEnv, get_successor_state
from pddlgym.inference import check_goal
from pddlgym.structs import Literal, LiteralConjunction, State


class PddlFileSimulator:
    """File-based PDDL simulator wrapping PDDLGym's PDDLEnv."""

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        use_vlm_heuristic: bool = False,
        use_vlm_action_selection: bool = False,
        chosen_goal=None,
    ):
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.chosen_goal = chosen_goal
        self._use_vlm_heuristic = use_vlm_heuristic
        self._use_vlm_action_selection = use_vlm_action_selection

        self._tmpdir = tempfile.mkdtemp(prefix="pddlsim_")
        problem_path = Path(problem_file)
        shutil.copy2(str(problem_path), os.path.join(self._tmpdir, problem_path.name))

        self.env = PDDLEnv(
            domain_file=domain_file,
            problem_dir=self._tmpdir,
            render=None,
            operators_as_actions=True,
            dynamic_action_space=True,
        )

        problem_name = problem_path.stem
        problem_idx = 0
        for i, p in enumerate(self.env.problems):
            if Path(p.problem_fname).stem == problem_name:
                problem_idx = i
                break
        self.env.fix_problem_index(problem_idx)

        self.state: Optional[State] = None
        self.domain = self.env.domain

    def __del__(self):
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def reset(self) -> State:
        self.state, _ = self.env.reset()
        if self.chosen_goal is not None:
            if isinstance(self.chosen_goal, (Literal, LiteralConjunction)):
                pass
            elif isinstance(self.chosen_goal, list):
                matched = [lit for lit in self.state.goal.literals if str(lit) in self.chosen_goal]
                if len(matched) != len(self.chosen_goal):
                    raise ValueError(
                        f"chosen_goal list contains literals not in problem goal: {self.chosen_goal}"
                    )
                self.chosen_goal = LiteralConjunction(matched)
            elif isinstance(self.chosen_goal, str):
                found = False
                for lit in self.state.goal.literals:
                    if str(lit) == self.chosen_goal:
                        self.chosen_goal = lit
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"chosen_goal string not found in problem goal literals: {self.chosen_goal}"
                    )
        else:
            self.chosen_goal = self.state.goal
        return self.state

    def step(self, state: State, action: Any) -> State:
        return get_successor_state(state, action, self.domain)

    def get_actions(self, state: State) -> List[Any]:
        actions = self.env.action_space.all_ground_literals(state)
        return sorted(actions, key=lambda x: str(x), reverse=True)

    def is_goal(self, state: State) -> bool:
        return check_goal(state, self.chosen_goal)

    def state_to_atoms(self, state: State) -> Dict[str, Set[Literal]]:
        return {"atoms": set(state.literals)}

    def atoms_to_state(self, atoms: Dict[str, Set[Literal]]) -> State:
        if "atoms" not in atoms:
            raise ValueError("atoms dict must contain 'atoms' key")
        literals = frozenset(atoms["atoms"])
        objects = self.state.objects if self.state else frozenset()
        goal = self.state.goal if self.state else None
        return State(literals, objects, goal)

    def make_observation(self, state: State) -> Dict[str, Set[Literal]]:
        return self.state_to_atoms(state)

    def get_state_probability(self, state: State) -> Dict[Literal, float]:
        return {lit: 1.0 for lit in state.literals}
