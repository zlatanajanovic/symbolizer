#!/usr/bin/env python3
"""
Reproduce the paper tables (Tables 1-6) by running the shipped code.

This is a **code-only** release: it ships the experiment code and all the input
data needed to reproduce every table, but it does **not** ship our precomputed
result files. You reproduce the numbers by running the experiments yourself.

There are two kinds of table:

  * Offline symbolic-planning tables (Tables 4-5, and the symbolic ViPlan rows of
    Table 6) need no API and no cost. This script re-executes the shipped
    symbolic planner on the bundled problems and reports solved / total, which is
    exactly the planning-success reproduction. This runs by default.

  * Grounding / VLM-in-the-loop tables (Tables 1-3, the VLM rows of Tables 5-6)
    make live VLM calls. This script does not launch paid calls automatically;
    `--print-live-commands` prints the exact, ready-to-run recipe for each.

Compare the numbers you obtain against the values reported in the paper.

Usage:
    python -m experiments.regenerate_all_tables                  # offline planner reproduction
    python -m experiments.regenerate_all_tables --smoke          # fast offline planner re-exec (budget 2000)
    python -m experiments.regenerate_all_tables --budget 10000   # offline planner, custom budget
    python -m experiments.regenerate_all_tables --print-live-commands
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def verify_planner(budget):
    """Re-execute the shipped symbolic planner on the bundled problems (offline, no API).

    This reproduces the offline planning-success numbers (Tables 4-5 and the
    symbolic ViPlan rows of Table 6): the shipped A* / A*+Novelty search over the
    checked-in PDDL problems. Returns True iff the planner solved problems for
    both A* columns.
    """
    print(f"\n{'='*100}\nPLANNER RE-EXECUTION (shipped code, offline, budget={budget})\n{'='*100}")
    out = os.path.join(RESULTS, "generated", "planner_verify")
    cmd = [sys.executable, "-m", "experiments.run_planning_benchmark",
           "--data_dir", os.path.join(HERE, "data", "planning"),
           "--budget", str(budget), "--output_dir", out]
    r = subprocess.run(cmd, cwd=os.path.dirname(HERE), capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("PLANNER RE-EXEC FAILED:\n", r.stderr[-1500:])
        return False
    try:
        payload = json.load(open(os.path.join(out, "planning_benchmark_results.json")))
        gt = payload.get("grand_totals", {})
        ok = True
        for alg in ("A*", "A*+Nov"):
            s = gt.get(alg, {})
            print(f"  paper-method column {alg:7s}: solved {s.get('solved')}/{s.get('total')}")
            if not s.get("solved"):
                ok = False
        return ok
    except Exception as e:  # noqa: BLE001
        print("could not parse planner output:", e)
        return False


LIVE_COMMANDS = """
=== LIVE (from-scratch) re-run recipes — call paid VLM APIs ===

Prereqs: configure .env (MODEL_TO_USE + credentials). Gemini via Vertex needs
GOOGLE_APPLICATION_CREDENTIALS; Mistral runs cost money (do not re-run casually).

Tables 1-3  (grounding F1 — objects / predicates / goal):
    python -m experiments.run_grounding_eval --config config.yaml --dataset pddlgym
    python -m experiments.run_grounding_eval --config config.yaml --dataset real
    python -m experiments.run_grounding_eval --config config.yaml --dataset vilain

Tables 4-5  (planning success):
    # Offline symbolic search over the checked-in problems (no API, no cost):
    python -m experiments.run_planning_benchmark --data_dir experiments/data/planning
    # VLM-grounded planning (VLM in the loop), live:
    python -m experiments.run_search_comparison --env PDDLEnvHanoi-v0 --problem_index 0

Table 6  (ViPlan):
    # Symbolic ViPlan planning (needs an external ViPlan checkout: export VIPLAN_ROOT=/path/to/ViPlan):
    python -m experiments.run_planning_benchmark --data_dir experiments/data/planning
    # Live ViPlan Blocksworld render (Blender) + VLM grounding on the rendered image:
    python examples/viplan_blocksworld_vlm_example.py

Note: a fresh live re-run is a NEW sample and may differ slightly from the paper's
printed values due to VLM nondeterminism. Compare against the values reported in
the paper; treat live re-runs as independent confirmation that the pipeline runs.
""".strip()


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the paper tables by running the shipped code "
                    "(offline planner reproduction + live re-run recipes).")
    ap.add_argument("--verify-planner", action="store_true",
                    help="Re-execute the shipped symbolic planner on bundled problems (offline). Default action.")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast offline planner re-exec (budget 2000). No paid calls.")
    ap.add_argument("--budget", type=int, default=5000, help="Planner expansion budget.")
    ap.add_argument("--print-live-commands", action="store_true",
                    help="Print the live from-scratch re-run recipe for each table and exit.")
    args = ap.parse_args()

    if args.print_live_commands:
        print(LIVE_COMMANDS)
        return 0

    print(f"{'='*100}")
    print("Code-only release: result files are not shipped — reproduce them by running the code.")
    print("  * Offline planning-success tables (4-5, symbolic ViPlan rows of 6) reproduce below.")
    print("  * Live grounding / VLM tables (1-3, VLM rows of 5-6): see --print-live-commands.")
    print("Compare the numbers you obtain against the values reported in the paper.")
    print(f"{'='*100}")

    # The offline planner reproduction always runs (it needs no result files).
    planner_ok = verify_planner(budget=2000 if args.smoke else args.budget)

    print(f"\nRESULT: {'OK' if planner_ok else 'FAIL'} (planner_ok={planner_ok})")
    return 0 if planner_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
