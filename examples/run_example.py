#!/usr/bin/env python3
"""
Minimal end-to-end example: image -> ground (objects, predicates, goal) -> PDDL -> A* plan.

This runs the REAL SYMBOLIZER pipeline on a bundled grounding-dataset state:
  1. Load a grounding dataset and pick one problem state (image + domain + goal text)
  2. VLM-ground objects, predicates, and the goal via structured JSON output
  3. Assemble a PDDL problem from the grounded JSON
  4. Solve it with A* + goal-count heuristic (the paper's planning method)

It makes live VLM calls — configure .env (or pass provider env vars) first.

Usage:
    # default: PDDLGym hanoi, problem 0, 1-shot
    python examples/run_example.py

    python examples/run_example.py --dataset pddlgym --domain_name hanoi --problem 0
    python examples/run_example.py --dataset real --domain_name blocksworld_real --num_examples 1
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DATASETS = {
    "pddlgym": "./experiments/data/datasets/pddlGYM_dataset_grounding.json",
    "vilain":  "./experiments/data/datasets/vilaln_data.json",
    "real":    "./experiments/data/datasets/RealImageDataset.json",
}


def _domain_pddl_name(domain_file):
    txt = Path(domain_file).read_text()
    m = re.search(r"\(define\s*\(domain\s+([^\s\)]+)", txt, re.I)
    return m.group(1) if m else Path(domain_file).stem


def _args_of(d):
    return [d[k]["name"] for k in sorted(d)
            if k != "predicate_type" and isinstance(d[k], dict) and "name" in d[k]]


def _pred_str(d):
    pred = d.get("predicate_type")
    if not pred:
        return None
    args = _args_of(d)
    return f"({pred} {' '.join(args)})" if args else f"({pred})"


def build_problem_pddl(domain_file, objects, atoms, goals, problem_name="grounded_problem"):
    """Assemble a PDDL problem string from grounded objects/atoms/goal dicts.

    Mirrors symbolizer.ssr_parsing.pddl_from_json for a single grounded instance.
    """
    type_map = {}
    for o in objects:
        if o.get("name") and o.get("type"):
            type_map.setdefault(o["type"], []).append(o["name"])
    objects_str = "\n    ".join(" ".join(v) + " - " + t for t, v in type_map.items())

    init_str = "\n    ".join(s for s in (_pred_str(a) for a in atoms) if s)

    gpreds = [s for s in (_pred_str(g) for g in goals) if s]
    if not gpreds:
        goal_str = "()"
    elif len(gpreds) == 1:
        goal_str = gpreds[0]
    else:
        goal_str = "(and " + " ".join(gpreds) + ")"

    content = (
        f"(define (problem {problem_name})\n"
        f"  (:domain {_domain_pddl_name(domain_file)})\n"
        f"  (:objects\n    {objects_str})\n"
        f"  (:init\n    {init_str})\n"
        f"  (:goal\n    {goal_str}))"
    )
    return content.lower()


def main():
    ap = argparse.ArgumentParser(description="SYMBOLIZER: image -> grounding -> PDDL -> A* plan")
    ap.add_argument("--dataset", choices=list(DATASETS), default="pddlgym")
    ap.add_argument("--domain_name", default="hanoi",
                    help="Domain name within the dataset to ground")
    ap.add_argument("--problem", type=int, default=0,
                    help="Which problem (index within the chosen domain) to use")
    ap.add_argument("--num_examples", type=int, default=1,
                    help="Few-shot examples for grounding (0 = zero-shot)")
    ap.add_argument("--budget", type=int, default=5000, help="A* expansion budget")
    ap.add_argument("--config", default="./config.yaml")
    args = ap.parse_args()

    print("=" * 64)
    print("SYMBOLIZER — End-to-End: image -> grounding -> PDDL -> A* plan")
    print("=" * 64)

    from symbolizer.ssr_parsing.parsing_utils.general_utils import load_env, load_config
    from symbolizer.ssr_parsing.run_eval_vlm import reinitialize_client
    from symbolizer.ssr_parsing.ssr_parsing import retrieve_objects, retrieve_predicates, retrieve_goal

    config = load_config(args.config)
    load_env(config["env_file"])
    reinitialize_client()
    print(f"Model: MODEL_TO_USE={os.getenv('MODEL_TO_USE')} MODEL_NAME={os.getenv('MODEL_NAME')}")

    dataset_path = DATASETS[args.dataset]
    dataset = json.load(open(dataset_path))

    # Resolve the target problem: the args.problem-th problem with this domain_name.
    matches = [(i, p) for i, p in enumerate(dataset["problems"]) if p["domain_name"] == args.domain_name]
    if not matches:
        sys.exit(f"No problems with domain_name={args.domain_name} in {dataset_path}")
    problem_id, prob = matches[min(args.problem, len(matches) - 1)]
    state = prob["states"][0]
    image_path = state["image_path"]
    domain_file = prob["domain_file"]
    print(f"Dataset: {args.dataset}  domain: {args.domain_name}  problem_id: {problem_id}")
    print(f"Image:   {image_path}")
    print(f"Domain:  {domain_file}")

    instruction_text = None
    gip = state.get("goal_instruction_path")
    if gip and os.path.exists(gip):
        instruction_text = Path(gip).read_text()

    # --- Step 1: objects ---
    print("\n--- Step 1: Object grounding ---")
    obj_list, _ = retrieve_objects(
        image_path=image_path, domain_name=args.domain_name, dataset=dataset,
        num_examples=args.num_examples, dataset_path=dataset_path,
        number_answers=1, problem_id=problem_id, instruction_text=instruction_text,
    )
    if not obj_list:
        raise SystemExit("Grounding failed: the VLM returned no parseable objects "
                         "(check your API key / model / image and retry).")
    objects = obj_list[0].dict().get("objects", [])
    print(f"  grounded {len(objects)} objects: " + ", ".join(o.get("name", "?") for o in objects))

    # --- Step 2: predicates ---
    print("\n--- Step 2: Predicate grounding ---")
    atom_list, _ = retrieve_predicates(
        image_path=image_path, domain_name=args.domain_name, objects=obj_list[0], dataset=dataset,
        num_examples=args.num_examples, dataset_path=dataset_path,
        number_answers=1, problem_id=problem_id, instruction_text=instruction_text,
    )
    if not atom_list:
        raise SystemExit("Grounding failed: the VLM returned no parseable predicates.")
    atoms = atom_list[0].dict().get("grounded_predicates", [])
    print(f"  grounded {len(atoms)} init atoms")

    # --- Step 3: goal ---
    print("\n--- Step 3: Goal grounding ---")
    if instruction_text and len(instruction_text.strip()) > 5:
        goal_list, _ = retrieve_goal(
            instruction_text=instruction_text, domain_name=args.domain_name, objects=obj_list[0],
            dataset=dataset, num_examples=args.num_examples, dataset_path=dataset_path,
            number_answers=1, problem_id=problem_id,
        )
        if not goal_list:
            raise SystemExit("Goal grounding failed: the VLM returned no parseable goal predicates.")
        goals = goal_list[0].dict().get("grounded_predicates", [])
        print(f"  grounded goal from instruction: {len(goals)} predicates")
    else:
        goals = json.loads(state["goal_predicates"]).get("grounded_predicates", []) if state.get("goal_predicates") else []
        print(f"  no instruction text; using dataset goal: {len(goals)} predicates")

    # --- Step 4: assemble PDDL problem ---
    print("\n--- Step 4: PDDL problem from grounding ---")
    problem_pddl = build_problem_pddl(domain_file, objects, atoms, goals)
    print(problem_pddl[:500] + ("..." if len(problem_pddl) > 500 else ""))

    # --- Step 5: plan with A* + goal-count heuristic ---
    print("\n--- Step 5: A* plan (goal-count heuristic) ---")
    from experiments.run_planning_benchmark import PyperplanSim, WithGoalCountH
    from symbolizer.search_eval.search.iw import AStarLLM

    tmpdir = tempfile.mkdtemp(prefix="symbolizer_example_")
    prob_file = os.path.join(tmpdir, "grounded_problem.pddl")
    Path(prob_file).write_text(problem_pddl)

    def _plan(dfile, pfile):
        sim = WithGoalCountH(PyperplanSim(dfile, pfile))
        init = sim.reset()
        # search() returns [] both when the initial state already satisfies the
        # goal AND when no solution exists, so decide success explicitly: an
        # already-satisfied goal is a valid 0-step plan, not a failure.
        if sim.is_goal(init):
            return [], 0, True
        plan, expansions = AStarLLM(max_num_expansions=args.budget).search(sim, init)
        return plan, expansions, bool(plan)

    plan, expansions, solved = None, 0, False
    try:
        plan, expansions, solved = _plan(domain_file, prob_file)
    except Exception as e:  # noqa: BLE001
        print(f"  grounded problem did not parse/plan ({type(e).__name__}: {str(e)[:80]})")

    if solved and plan:
        print(f"  PLAN FOUND on the VLM-grounded problem ({len(plan)} steps, {expansions} expansions):")
        for i, a in enumerate(plan):
            print(f"    {i+1}. {a}")
    elif solved:
        print("  PLAN FOUND: the VLM-grounded initial state already satisfies the goal (0-step plan).")
    else:
        # No fallback. We plan the REAL VLM-grounded problem and report the result
        # as-is. An imperfect grounding (F1 < 1) can yield an unsolvable instance,
        # and grounding-only domains (e.g. ViLaIn cooking, which is :adl/:functions)
        # are outside the STRIPS A* planner's scope -- both are surfaced honestly
        # rather than masked by re-planning a different reference problem.
        print(f"  no plan found on the VLM-grounded problem (expansions={expansions}).")

    print("\n" + "=" * 64)
    print("Done — grounding ran live; A* planning ran on the symbolic problem.")
    print("(Full VLM-grounded planning benchmark: experiments/run_planning_benchmark.py)")
    print("=" * 64)
    return 0 if solved else 2


if __name__ == "__main__":
    raise SystemExit(main())
