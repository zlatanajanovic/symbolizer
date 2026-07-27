#!/usr/bin/env python3
"""
Reproduce planning benchmark results (Tables 4-5 in the paper).

Evaluates search algorithms (IWK, SIW, BFWS, A*, A*+Novelty) on
symbolic PDDL problems across multiple domains:
  - PDDLGym: Blocksworld, Hanoi, Hanoi Color
  - PyBullet: Blocksworld, Hanoi
  - Kitchen-Worlds
  - ViPlan: Blocksworld, Household

Uses pyperplan for domain-independent PDDL parsing/grounding.

Usage:
    python -m experiments.run_planning_benchmark
    python -m experiments.run_planning_benchmark --budget 10000
    python -m experiments.run_planning_benchmark --data_dir ./experiments/data/planning
"""

import sys
import os
import argparse
import heapq
import json
from itertools import combinations
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# scipy compatibility fix for ConfigSpace/unified_planning
import scipy.stats._distn_infrastructure as _scipy_di
if not hasattr(_scipy_di, "rv_continuous_frozen"):
    _scipy_di.rv_continuous_frozen = _scipy_di.rv_frozen

from pyperplan import grounding
from pyperplan.pddl.parser import Parser as PddlParser

from symbolizer.search_eval.search.iw import IWK, SIW, BFWS, AStarLLM
from symbolizer.search_eval.simulator.pddl_file_simulator import PddlFileSimulator
from symbolizer.search_eval.simulator.viplan_blocksworld_simulator import (
    ViPlanBlocksworldSimulator,
    VIPLAN_AVAILABLE,
)


# ---------------------------------------------------------------------------
# pyperplan wrapper
# ---------------------------------------------------------------------------

class GoalWrapper:
    """Makes a frozenset of goal strings look like pddlgym's LiteralConjunction."""
    def __init__(self, goals):
        self.literals = list(goals)


class PyperplanSim:
    """Domain-independent simulator using pyperplan for PDDL parsing/grounding."""
    def __init__(self, domain_file, problem_file):
        parser = PddlParser(str(domain_file), str(problem_file))
        domain = parser.parse_domain()
        self.task = grounding.ground(parser.parse_problem(domain))
        self.chosen_goal = GoalWrapper(self.task.goals)

    def reset(self):
        return self.task.initial_state

    def step(self, state, action):
        return action.apply(state)

    def get_actions(self, state):
        return [op for op in self.task.operators if op.applicable(state)]

    def is_goal(self, state):
        return self.task.goals.issubset(state)

    def make_observation(self, state):
        return {"atoms": state}


# Fix AStarLLM state key to handle frozenset states (pyperplan)
def _generic_state_key(self, state):
    if hasattr(state, "literals"):
        return frozenset(state.literals)
    if hasattr(state, "atoms"):
        return frozenset(state.atoms)
    return frozenset(state)

AStarLLM._state_key = _generic_state_key


# ---------------------------------------------------------------------------
# Goal-count heuristic wrapper
# ---------------------------------------------------------------------------

class WithGoalCountH:
    """Wraps a simulator to add goal-count heuristic scoring."""
    def __init__(self, sim):
        self._sim = sim

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def score_state_with_heuristic(self, state):
        goal = getattr(self._sim, "chosen_goal", None)
        if goal is None:
            goal = getattr(state, "goal", None)

        if hasattr(goal, "literals"):
            goal_strs = {str(l) for l in goal.literals}
        elif isinstance(goal, (frozenset, set)):
            goal_strs = {str(g) for g in goal}
        else:
            goal_strs = set()

        if hasattr(state, "literals"):
            state_strs = {str(l) for l in state.literals}
        elif hasattr(state, "atoms"):
            state_strs = {str(a) for a in state.atoms}
        else:
            state_strs = {str(a) for a in state}

        total = len(goal_strs)
        if total == 0:
            return {"heuristic_score": 10}
        sat = len(goal_strs & state_strs)
        return {"heuristic_score": int(10 * sat / total)}


class ViPlanWithGoalH:
    """Wraps ViPlanBlocksworldSimulator with goal-count heuristic."""
    def __init__(self, sim):
        self._sim = sim
        self._init_state = sim.reset()
        self._goal_strs = frozenset(str(g) for g in self._init_state.goal)
        self.chosen_goal = GoalWrapper(self._goal_strs)

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def reset(self):
        return self._init_state

    def score_state_with_heuristic(self, state):
        state_strs = frozenset(str(a) for a in state.atoms)
        total = len(self._goal_strs)
        if total == 0:
            return {"heuristic_score": 10}
        sat = len(self._goal_strs & state_strs)
        return {"heuristic_score": int(10 * sat / total)}


# ---------------------------------------------------------------------------
# A* with novelty tiebreaker
# ---------------------------------------------------------------------------

class AStarNovelty:
    """A* (goal-count h) with novelty as tiebreaker.

    Priority: (g+h, novelty_rank, counter)
      novelty_rank=1: state has a new single atom not seen before
      novelty_rank=2: state has a new atom-pair not seen before
      novelty_rank=3: state is fully redundant
    """

    def __init__(self, max_num_expansions=50_000, max_width=2):
        self.max_num_expansions = max_num_expansions
        self.max_width = max_width

    def search(self, simulator, initial_state):
        goal = getattr(simulator, "chosen_goal", None)
        if goal is None:
            goal = getattr(initial_state, "goal", None)
        if hasattr(goal, "literals"):
            goal_strs = frozenset(str(l) for l in goal.literals)
        elif isinstance(goal, (frozenset, set)):
            goal_strs = frozenset(str(g) for g in goal)
        else:
            goal_strs = frozenset()

        def get_atoms(state):
            if hasattr(state, "literals"):
                return frozenset(str(l) for l in state.literals)
            if hasattr(state, "atoms"):
                return frozenset(str(a) for a in state.atoms)
            return frozenset(str(a) for a in state)

        def state_key(state):
            if hasattr(state, "literals"):
                return frozenset(state.literals)
            if hasattr(state, "atoms"):
                return frozenset(state.atoms)
            return frozenset(state)

        def novelty_rank(atoms_t, table):
            for w in range(1, self.max_width + 1):
                for combo in combinations(atoms_t, w):
                    if table.get(combo, 0) == 0:
                        return w
            return self.max_width + 1

        def update_table(atoms_t, table):
            for w in range(1, self.max_width + 1):
                for combo in combinations(atoms_t, w):
                    table[combo] = table.get(combo, 0) + 1

        table = {}
        visited = {}
        heap = []
        counter = 0

        init_atoms = get_atoms(initial_state)
        init_atoms_t = tuple(sorted(init_atoms))
        init_h = len(goal_strs - init_atoms)
        init_nov = novelty_rank(init_atoms_t, table)

        heapq.heappush(heap, (init_h, 0, init_nov, counter, initial_state, [], 0))
        counter += 1
        num_exp = 0

        while heap:
            f, _, nov, _, state, path, g = heapq.heappop(heap)

            sk = state_key(state)
            if sk in visited and visited[sk] <= g:
                continue
            visited[sk] = g

            num_exp += 1
            if num_exp > self.max_num_expansions:
                return [], num_exp

            atoms = get_atoms(state)
            atoms_t = tuple(sorted(atoms))
            update_table(atoms_t, table)

            if simulator.is_goal(state):
                return path, num_exp

            for action in simulator.get_actions(state):
                next_state = simulator.step(state, action)
                next_sk = state_key(next_state)
                next_g = g + 1
                if next_sk in visited and visited[next_sk] <= next_g:
                    continue
                next_atoms = get_atoms(next_state)
                next_h = len(goal_strs - next_atoms)
                next_nov = novelty_rank(tuple(sorted(next_atoms)), table)
                heapq.heappush(heap, (next_g + next_h, next_g, next_nov, counter,
                                      next_state, path + [action], next_g))
                counter += 1

        return [], num_exp


# ---------------------------------------------------------------------------
# Run one problem
# ---------------------------------------------------------------------------

def run_one(dom_path, prob_path, alg_name, backend="pyperplan", budget=50_000):
    try:
        if backend == "viplan_bw":
            raw_sim = ViPlanBlocksworldSimulator(str(dom_path), str(prob_path), state_source="symbolic")
            sim_factory = lambda: raw_sim
        elif backend == "pddlgym":
            sim_factory = lambda: PddlFileSimulator(str(dom_path), str(prob_path))
        else:
            sim_factory = lambda: PyperplanSim(dom_path, prob_path)

        if alg_name in ("A*", "A*+Nov"):
            base = sim_factory()
            sim = ViPlanWithGoalH(base) if backend == "viplan_bw" else WithGoalCountH(base)
            if alg_name == "A*":
                algo = AStarLLM(max_num_expansions=budget)
            else:
                algo = AStarNovelty(max_num_expansions=budget)
        else:
            sim = sim_factory()
            if alg_name == "IWK":
                algo = IWK(max_width=2, K=1, max_num_observations=budget)
            elif alg_name == "SIW":
                algo = SIW(max_width=2, K=1, max_num_observations=budget)
            elif alg_name == "BFWS":
                algo = BFWS(max_width=2, K=1, max_num_observations=budget)

        state = sim.reset()
        plan, expansions = algo.search(sim, state)
        solved = plan is not None and len(plan) > 0
        return solved, expansions, len(plan) if solved else 0
    except Exception:
        # Do NOT silently report a planner/parse error as a legitimate "unsolved"
        # — that would corrupt the reproduced success rates. Surface the full
        # traceback (loud) and continue with the remaining problems.
        import sys as _sys, traceback as _tb
        print(f"[run_one] ERROR on {dom_path} / {prob_path} "
              f"[{alg_name}/{backend}]:", file=_sys.stderr)
        _tb.print_exc()
        return False, 0, 0


def get_problems(base, rel_path, backend="pyperplan", cap=None):
    """Get list of problem directories under base/rel_path."""
    path = base / rel_path
    if not path.exists():
        return []
    dom_file = "domain_pddlgym.pddl" if backend == "pddlgym" else "domain.pddl"
    probs = sorted(
        p for p in path.iterdir()
        if p.is_dir() and (p / dom_file).exists() and (p / "problem.pddl").exists()
    )
    return probs[:cap] if cap else probs


def relative_path(path, base):
    """Return a stable path for JSON output."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def summarize_counts(solved, total_expansions, total):
    avg = total_expansions // solved if solved > 0 else 0
    return {
        "solved": int(solved),
        "total": int(total),
        "success_rate": float(solved / total) if total else 0.0,
        "total_expansions": int(total_expansions),
        "avg_expansions_per_solved": int(avg),
    }


def write_results(output_dir, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "planning_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Planning benchmark evaluation")
    parser.add_argument("--data_dir", type=str, default="./experiments/data/planning",
                        help="Path to planning data directory")
    parser.add_argument("--budget", type=int, default=50_000,
                        help="Max node expansions budget")
    parser.add_argument("--output_dir", type=str, default="./experiments/results/generated/planning",
                        help="Directory for JSON output")
    parser.add_argument("--dry_run", action="store_true",
                        help="List benchmark inputs and write JSON metadata without running search")
    args = parser.parse_args()

    DATA = Path(args.data_dir)
    BUDGET = args.budget
    OUTPUT_DIR = Path(args.output_dir)

    # (rel_path, label, backend, cap)
    DOMAINS = [
        ("pddlgym/blocks",         "PDDLGym-Blocks",    "pyperplan", None),
        ("pddlgym/hanoi",          "PDDLGym-Hanoi",      "pyperplan", None),
        ("pddlgym/hanoi_color",    "PDDLGym-HanoiColor", "pyperplan", None),
        ("pybullet/blocks",        "PyBullet-Blocks",    "pyperplan", None),
        ("pybullet/hanoi",         "PyBullet-Hanoi",     "pyperplan", None),
        ("kitchen/kitchen_worlds", "Kitchen",            "pyperplan", None),
        ("viplan/blocksworld",     "ViPlan-Blocks",      "viplan_bw", None),
        ("viplan/household",       "ViPlan-Household",   "pddlgym",   None),
    ]

    ALGOS = ["IWK", "SIW", "BFWS", "A*", "A*+Nov"]

    payload = {
        "data_dir": str(DATA),
        "budget": BUDGET,
        "dry_run": bool(args.dry_run),
        "algorithms": ALGOS,
        "domains": [],
        "grand_totals": {},
    }

    if args.dry_run:
        print()
        print("Planning benchmark dry run")
        print(f"Data directory: {DATA}")
        for rel_path, label, backend, cap in DOMAINS:
            if backend == "viplan_bw" and not VIPLAN_AVAILABLE:
                print(f"  {label:<20}   - (skipped — requires external ViPlan checkout; set VIPLAN_ROOT)")
                payload["domains"].append({
                    "label": label, "rel_path": rel_path, "backend": backend,
                    "n": 0, "skipped": "requires VIPLAN_ROOT", "problems": [], "summary": {},
                })
                continue
            problems = get_problems(DATA, rel_path, backend, cap=cap)
            print(f"  {label:<20} {len(problems):>3} problems")
            payload["domains"].append({
                "label": label,
                "rel_path": rel_path,
                "backend": backend,
                "n": len(problems),
                "problems": [relative_path(p, DATA) for p in problems],
                "summary": {},
            })
        write_results(OUTPUT_DIR, payload)
        return

    # Print header
    W = 17
    print()
    print(f"{'Domain':<22} {'N':>3}  " + "  ".join(f"{'  '+a+' (sol / avg_exp)':<{W}}" for a in ALGOS))
    print("-" * (22 + 3 + (W + 2) * len(ALGOS) + 5))

    grand = {a: [0, 0, 0] for a in ALGOS}

    for rel_path, label, backend, cap in DOMAINS:
        if backend == "viplan_bw" and not VIPLAN_AVAILABLE:
            print(f"  {label:<20}  (skipped — requires external ViPlan checkout; set VIPLAN_ROOT)", flush=True)
            payload["domains"].append({
                "label": label, "rel_path": rel_path, "backend": backend,
                "n": 0, "skipped": "requires VIPLAN_ROOT", "problems": [], "summary": {},
            })
            continue
        problems = get_problems(DATA, rel_path, backend, cap=cap)
        n = len(problems)
        domain_result = {
            "label": label,
            "rel_path": rel_path,
            "backend": backend,
            "n": n,
            "problems": [],
            "summary": {},
        }
        if n == 0:
            print(f"  {label:<20}  (no problems found)", flush=True)
            payload["domains"].append(domain_result)
            continue

        dom_totals = {a: [0, 0] for a in ALGOS}

        for prob_dir in problems:
            dom_file = "domain_pddlgym.pddl" if backend == "pddlgym" else "domain.pddl"
            problem_result = {
                "problem": relative_path(prob_dir, DATA),
                "domain_file": relative_path(prob_dir / dom_file, DATA),
                "problem_file": relative_path(prob_dir / "problem.pddl", DATA),
                "results": {},
            }
            for alg in ALGOS:
                solved, exp, plan_len = run_one(
                    prob_dir / dom_file, prob_dir / "problem.pddl",
                    alg, backend=backend, budget=BUDGET
                )
                dom_totals[alg][0] += int(solved)
                dom_totals[alg][1] += exp
                grand[alg][0] += int(solved)
                grand[alg][1] += exp
                grand[alg][2] += 1
                problem_result["results"][alg] = {
                    "solved": bool(solved),
                    "expansions": int(exp),
                    "plan_len": int(plan_len),
                }
            domain_result["problems"].append(problem_result)

        cells = []
        for alg in ALGOS:
            s, tot_exp = dom_totals[alg]
            avg = tot_exp // s if s > 0 else 0
            cells.append(f"{s:>2}/{n}  avg={avg:>5}")
            domain_result["summary"][alg] = summarize_counts(s, tot_exp, n)
        print(f"  {label:<20} {n:>3}  " + "  ".join(f"{c:<{W}}" for c in cells), flush=True)
        payload["domains"].append(domain_result)

    # Grand total
    print("-" * (22 + 3 + (W + 2) * len(ALGOS) + 5))
    cells = []
    for alg in ALGOS:
        s, tot_exp, total_n = grand[alg]
        avg = tot_exp // s if s > 0 else 0
        cells.append(f"{s:>3}/{total_n}  avg={avg:>5}")
        payload["grand_totals"][alg] = summarize_counts(s, tot_exp, total_n)
    print(f"  {'TOTAL (all probs)':<20} {'':>3}  " + "  ".join(f"{c:<{W}}" for c in cells))
    print()
    write_results(OUTPUT_DIR, payload)


if __name__ == "__main__":
    main()
