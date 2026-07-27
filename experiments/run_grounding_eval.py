#!/usr/bin/env python3
"""
Reproduce grounding evaluation results (Tables 1-3 in the paper).

Evaluates object, predicate, and goal grounding F1 scores using VLM
structured output on PDDLGym and real-image datasets.

Usage:
    python -m experiments.run_grounding_eval --config config.yaml
    python -m experiments.run_grounding_eval --config config.yaml --dataset pddlgym
    python -m experiments.run_grounding_eval --config config.yaml --dataset real
"""

import argparse
import sys
import os

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Run grounding evaluation (object/predicate/goal F1)"
    )
    parser.add_argument(
        "--config", type=str, default="./config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "pddlgym", "real", "vilain"],
        help="Which dataset to evaluate on"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/grounding",
        help="Directory for output results"
    )
    parser.add_argument(
        "--domain_name", type=str, default=None,
        help="Domain filter to pass to the grounding evaluator"
    )
    parser.add_argument(
        "--num_tests", type=int, default=None,
        help="Override num_tests from config"
    )
    parser.add_argument(
        "--oracle_stage", type=str, default="none",
        choices=["none", "objects", "objects_predicates"],
        help="Cascade oracle stage: none, objects, or objects_predicates"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate selected work without making API calls"
    )
    args = parser.parse_args()

    # The core evaluation is in run_eval_on_dataset.py which handles:
    #   1. Loading dataset (PDDLGym or real images)
    #   2. Querying VLM for objects, predicates, goals via structured JSON
    #   3. Computing precision/recall/F1 against ground truth
    #
    # To run, set up your .env file with API keys and adjust config.yaml
    # to point to the desired dataset path.

    # Override config dataset path based on --dataset flag
    dataset_map = {
        "pddlgym": "./experiments/data/datasets/pddlGYM_dataset_grounding.json",
        "real":    "./experiments/data/datasets/RealImageDataset.json",
        "vilain":  "./experiments/data/datasets/vilaln_data.json",
    }

    # "all" means evaluate every grounding dataset in turn (each into its own
    # sub-directory), not "fall back to whatever config.yaml points at". Run each
    # as a fresh subprocess so the delegated evaluator re-reads the dataset
    # override cleanly (it parses argv / env at import time).
    if args.dataset == "all":
        import subprocess
        for ds in dataset_map:
            out_dir = os.path.join(args.output_dir, ds)
            cmd = [sys.executable, "-m", "experiments.run_grounding_eval",
                   "--config", args.config, "--dataset", ds,
                   "--output_dir", out_dir, "--oracle_stage", args.oracle_stage]
            if args.domain_name:
                cmd += ["--domain_name", args.domain_name]
            if args.num_tests is not None:
                cmd += ["--num_tests", str(args.num_tests)]
            if args.dry_run:
                cmd.append("--dry-run")
            print(f"\n=== Grounding eval: dataset={ds} -> {out_dir} ===", flush=True)
            subprocess.run(cmd, check=True)
        return

    # Single dataset: in-process delegation (unchanged behavior).
    os.environ["SYMBOLIZER_DATASET_OVERRIDE"] = dataset_map[args.dataset]
    os.environ["SYMBOLIZER_OUTPUT_OVERRIDE"] = args.output_dir
    if args.domain_name:
        os.environ["SYMBOLIZER_DOMAIN_OVERRIDE"] = args.domain_name
    if args.num_tests is not None:
        os.environ["SYMBOLIZER_NUM_TESTS_OVERRIDE"] = str(args.num_tests)

    # Delegate to the main evaluation script
    sys.argv = [
        "run_eval_on_dataset",
        "--config", args.config,
        "--oracle_stage", args.oracle_stage,
    ]
    if args.dry_run:
        sys.argv.append("--dry-run")
    from symbolizer.ssr_parsing.run_eval_on_dataset import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
