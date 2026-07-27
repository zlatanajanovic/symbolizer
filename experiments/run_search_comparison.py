#!/usr/bin/env python3
"""
VLM-grounded A* planning (the VLM in the planning loop).

For each state visited during search the VLM renders and grounds the PDDLGym
image (objects + predicates); A* then searches over the *true* environment
transitions, guided by a VLM goal-count heuristic and VLM action ordering. This
is the "Symbolizer + VLM" planning path.

Usage:
    python -m experiments.run_search_comparison \\
        --env PDDLEnvHanoi-v0 --problem_index 0 \\
        --max_vlm_obs 15 --max_astar_exp 60 \\
        --model gemini-3.1-flash-lite-preview

Each observation renders a high-res frame and calls the VLM twice; run on a host
with several GB of free RAM.
"""

import argparse
import json
import logging
import os
import sys
import time

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from symbolizer.ssr_parsing.run_eval_vlm import reinitialize_client
from symbolizer.search_eval.search.iw import AStarLLM
from symbolizer.ssr_parsing.ssr_parsing import select_best_action

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("search_comparison")


def _state_key(state):
    """Stable hashable key for a state across the various state representations."""
    if hasattr(state, "_key"):
        return state._key
    if hasattr(state, "literals"):
        return tuple(sorted(str(l) for l in state.literals))
    return tuple(sorted(str(x) for x in state))


# A* compares states by this key.
AStarLLM._state_key = staticmethod(_state_key)


def _goal_satisfied_count(goal_literals, obs_atom_strs):
    """Count satisfied goal literals (handles Anti/Not/negative goals)."""
    sat = 0
    for g in goal_literals:
        gs = str(g)
        is_neg = getattr(g, "is_negative", False) or getattr(g, "is_anti", False)
        if gs.startswith("Anti"):
            positive_form = gs[4:]
            if positive_form not in obs_atom_strs:
                sat += 1
        elif gs.startswith("Not") and not gs.startswith("Nothing"):
            positive_form = gs[3:]
            if positive_form and positive_form[0].isupper():
                positive_form = positive_form[0].lower() + positive_form[1:]
            if positive_form not in obs_atom_strs:
                sat += 1
        elif is_neg:
            pred = getattr(g, "predicate", None)
            if pred:
                pred_name = str(pred).lower()
                found = any(pred_name in a.lower() for a in obs_atom_strs)
                if not found:
                    sat += 1
            else:
                sat += 1
        else:
            if gs in obs_atom_strs:
                sat += 1
    return sat


class VLMSearchWrapper:
    """Full VLM wrapper providing observation, heuristic, and action selection."""

    def __init__(self, sim, max_vlm_obs=15):
        self._sim = sim
        self._obs_cache = {}
        self.vlm_obs_calls = 0
        self.vlm_action_calls = 0
        self.vlm_heuristic_calls = 0
        self._max_vlm_obs = max_vlm_obs
        self._cumulative_atoms = set()

    def __getattr__(self, n):
        return getattr(self._sim, n)

    def _vlm_atoms(self, state):
        key = _state_key(state)
        if key in self._obs_cache:
            return self._obs_cache[key]
        if self.vlm_obs_calls >= self._max_vlm_obs:
            return self._cumulative_atoms
        obs = self._sim.make_observation(state)
        new_atoms = {str(a) for a in obs.get("atoms", set())}
        self.vlm_obs_calls += 1
        self._cumulative_atoms |= new_atoms
        self._obs_cache[key] = set(self._cumulative_atoms)
        return self._obs_cache[key]

    def make_observation(self, state):
        atoms = self._vlm_atoms(state)
        return {"atoms": atoms}

    def score_state_with_heuristic(self, state):
        goal = getattr(self._sim, "chosen_goal", None) or getattr(state, "goal", None)
        if not goal or not hasattr(goal, "literals"):
            return {"heuristic_score": 1}
        obs_atoms = self._vlm_atoms(state)
        total = len(goal.literals)
        if not total:
            return {"heuristic_score": 10}
        sat = _goal_satisfied_count(goal.literals, obs_atoms)
        score = int(10 * sat / total)
        return {"heuristic_score": score}

    def choose_best_action(self, state, actions):
        action_strs = [str(a) for a in actions]
        action_lookup = {str(a): a for a in actions}
        state_dict = {"grounded_predicates": sorted(self._cumulative_atoms)}
        goal = getattr(self._sim, "chosen_goal", None)
        domain_name = getattr(self._sim, "_ssr_domain_name", None) or "household"
        self.vlm_action_calls += 1
        try:
            result = select_best_action(
                actions=action_strs,
                current_state=state_dict,
                goal=goal,
                domain_name=domain_name,
            )
        except Exception as e:
            log.warning("choose_best_action failed: %s", e)
            return actions[0] if actions else None
        if result and isinstance(result, dict):
            best = result.get("best_action")
            if best in action_lookup:
                return action_lookup[best]
        return actions[0] if actions else None


def main():
    parser = argparse.ArgumentParser(
        description="VLM-grounded A* planning: the VLM grounds each rendered PDDLGym "
                    "state during search (objects+predicates), A* searches over the "
                    "true transitions guided by the VLM goal-count heuristic.")
    parser.add_argument("--env", type=str, default="PDDLEnvHanoi-v0",
                        help="PDDLGym env name (e.g. PDDLEnvHanoi-v0, PDDLEnvBlocks-v0)")
    parser.add_argument("--dataset", type=str,
                        default="./experiments/data/datasets/pddlGYM_dataset_grounding.json",
                        help="Grounding dataset providing few-shot examples for the env's domain")
    parser.add_argument("--problem_index", type=int, default=0, help="Problem index within the env")
    parser.add_argument("--max_vlm_obs", type=int, default=15, help="Max VLM observation budget")
    parser.add_argument("--max_astar_exp", type=int, default=60, help="Max A* expansions")
    parser.add_argument("--model", type=str, default=None, help="VLM model (overrides .env MODEL_NAME)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Explicit --model wins; otherwise honor an existing .env MODEL_NAME; else
    # fall back to the documented default. (setdefault alone silently ignored
    # --model whenever .env already defined MODEL_NAME.)
    model_name = args.model or os.environ.get("MODEL_NAME") or "gemini-3.1-flash-lite-preview"
    os.environ["MODEL_NAME"] = model_name
    os.environ.setdefault("MODEL_TO_USE", "gemini")
    reinitialize_client()

    # Imported here (not at module top) because it pulls in pddlgym rendering deps.
    import tempfile
    from symbolizer.search_eval.simulator.pddlgym_openai_simulator import PDDLGymSimulatorOpenai

    print(f"Env: {args.env}  problem: {args.problem_index}  model: {model_name}")
    print(f"Max VLM obs: {args.max_vlm_obs}  Max A* expansions: {args.max_astar_exp}")
    print("NOTE: each observation renders the state and calls the VLM twice (objects + "
          "predicates) on a high-res frame; run on a host with several GB free RAM.")

    img_dir = tempfile.mkdtemp(prefix="vlm_search_obs_")
    base = PDDLGymSimulatorOpenai(
        args.env, args.dataset, img_dir, args.dataset,
        num_examples=1, problem_index=args.problem_index,
    )
    sim = VLMSearchWrapper(base, max_vlm_obs=args.max_vlm_obs)
    start = sim.reset()
    print(f"Goal: {base.chosen_goal}")

    t0 = time.time()
    plan, expansions = AStarLLM(max_num_expansions=args.max_astar_exp).search(sim, start)
    elapsed = time.time() - t0

    result = {
        "env": args.env, "problem_index": args.problem_index, "model": args.model,
        "solved": bool(plan), "plan_length": len(plan) if plan else 0,
        "expansions": expansions, "vlm_obs_calls": sim.vlm_obs_calls,
        "elapsed_sec": round(elapsed, 1),
        "plan": [str(a) for a in plan] if plan else None,
    }
    print("\n=== VLM-grounded A* result ===")
    print(f"  solved={result['solved']} plan_len={result['plan_length']} "
          f"expansions={expansions} vlm_obs_calls={sim.vlm_obs_calls} ({result['elapsed_sec']}s)")
    if plan:
        for i, a in enumerate(plan):
            print(f"    {i+1}. {a}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.output}")
    return 0 if plan else 2


if __name__ == "__main__":
    raise SystemExit(main())
