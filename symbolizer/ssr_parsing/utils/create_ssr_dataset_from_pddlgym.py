#!/usr/bin/env python3
"""
create_ssr_dataset_from_pddlgym.py
----------------------------------
DEPRECATED / UNMAINTAINED helper. Its default paths
(`experiments/ssr_eval/pddlGymExperiment/data`, `experiments/exp_datasets/`) do
**not** exist in this release and it is not referenced by any reproduction
recipe. The maintained builder for the paper's grounding dataset is
`experiments/build_pddlgym_grounding_dataset.py` (use that instead). Kept only
for reference; pass explicit paths if you run it.

Builds an SSR-compatible grounding dataset (objects, predicates, optional goals)
directly from the PDDLGym-style assets stored under
`experiments/ssr_eval/pddlGymExperiment/data`.

Each domain is expected to expose:
  data/<domain>/
    domain.pddl
    problems/problemX.pddl
    observations/problemX.(png|jpg)
    problemX.txt (optional goal instruction)

The resulting JSON matches what `ssr_parsing.run_eval_vlm` expects:
{
  "problems": [
     {
        "problem_name": ...,
        "domain_name": ...,
        "domain_file": ".../domain.pddl",
        "problem_file": ".../problems/problem1.pddl",
        "states": [
            {
              "id": 0,
              "image_path": ".../observations/problem1.jpg",
              "goal_instruction_path": ".../problem1.txt" or null,
              "goal_predicates": "<json string>",
              "pickle_path_observation": "",
              "all_objects": "<json string>",
              "atoms": "<json string>",
              "atoms_schema": {...},
              "actions": [],
              "timestamp": "...",
            }
        ]
     }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure the package directory is importable when executed as a script
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../symbolizer/symbolizer
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

try:
    from ssr_parsing.parsing_utils.generate_scenes_utils import (
        get_goal_from_problem,
        get_original_objects_and_atoms_from_problem,
    )
    from ssr_parsing.parsing_utils.chatgpt_utils import (
        PDDLDomainWrapper,
        PDDLProblemWrapper,
    )
except ModuleNotFoundError as exc:
    if exc.name == "pddlgym":
        raise SystemExit(
            "pddlgym is required to parse the SSR datasets. "
            "Install it via `pip install pddlgym` inside the conda/env used for experiments."
        ) from exc
    raise

DEFAULT_DATA_ROOT = Path("experiments/ssr_eval/pddlGymExperiment/data")
DEFAULT_OUTPUT = Path("experiments/exp_datasets/ssr_dataset_generated.json")


def discover_domains(data_root: Path) -> List[str]:
    """Return all domain folder names inside the data root."""
    return sorted(
        entry.name
        for entry in data_root.iterdir()
        if entry.is_dir()
    )


def resolve_image_path(observations_dir: Path, problem_stem: str) -> Optional[Path]:
    """Try multiple extensions to locate the rendered observation."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = observations_dir / f"{problem_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_goal_instruction_path(domain_dir: Path, problem_stem: str) -> Optional[Path]:
    """Look for an optional text instruction next to the domain assets."""
    candidate = domain_dir / f"{problem_stem}.txt"
    return candidate if candidate.exists() else None


def serialize_state_payload(
    domain_file: Path,
    problem_file: Path,
    goal_instruction_path: Optional[Path],
) -> Tuple[str, str, Dict[str, Any], str, Optional[str]]:
    """Serialize objects, atoms, schemas, and goals for the given problem."""
    (
        all_objects,
        atoms,
        _predicate_classes,
        AtomsSchema,
        _generated_classes,
        _predicates_structured,
    ) = get_original_objects_and_atoms_from_problem(
        domain_file_path=str(domain_file),
        problem_file_path=str(problem_file),
    )
    goal_predicates = get_goal_from_problem(
        domain_file_path=str(domain_file),
        problem_file_path=str(problem_file),
    )

    serialized_objects = (
        all_objects.json()
        if hasattr(all_objects, "json")
        else json.dumps(all_objects)
    )
    serialized_atoms = (
        atoms.json()
        if hasattr(atoms, "json")
        else json.dumps(atoms)
    )
    atoms_schema = (
        AtomsSchema.schema()
        if hasattr(AtomsSchema, "schema")
        else {}
    )
    serialized_goal = (
        goal_predicates.json()
        if hasattr(goal_predicates, "json")
        else json.dumps(goal_predicates)
        if isinstance(goal_predicates, (dict, list))
        else str(goal_predicates)
    )

    goal_instruction_value = (
        str(goal_instruction_path) if goal_instruction_path is not None else None
    )

    return serialized_objects, serialized_atoms, atoms_schema, serialized_goal, goal_instruction_value


def build_problem_entry(
    domain_dir: Path,
    problem_path: Path,
    image_path: Path,
    goal_instruction_path: Optional[Path],
    dataset_timestamp: str,
) -> Dict[str, Any]:
    """Construct a dataset entry for a single (problem, observation) pair."""
    domain_file = domain_dir / "domain.pddl"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain file: {domain_file}")

    domain_wrapper = PDDLDomainWrapper(str(domain_file))
    problem_wrapper = PDDLProblemWrapper(str(problem_path), domain_wrapper)

    serialized_objects, serialized_atoms, atoms_schema, serialized_goal, goal_instruction_value = serialize_state_payload(
        domain_file, problem_path, goal_instruction_path
    )

    state_entry = {
        "id": 0,
        "image_path": str(image_path),
        "goal_instruction_path": goal_instruction_value,
        "goal_predicates": serialized_goal,
        "pickle_path_observation": "",
        "all_objects": serialized_objects,
        "atoms": serialized_atoms,
        "atoms_schema": atoms_schema,
        "actions": [],
        "timestamp": dataset_timestamp,
    }

    problem_entry = {
        "problem_name": problem_wrapper.problem.name,
        "domain_name": domain_wrapper.domain.name,
        "domain_file": str(domain_file),
        "problem_file": str(problem_path),
        "states": [state_entry],
    }
    return problem_entry


def generate_dataset(
    data_root: Path,
    output_path: Path,
    domains: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Iterate over every domain/problem pair and build the SSR dataset."""
    selected_domains = list(domains) if domains else discover_domains(data_root)
    dataset = {"problems": []}
    timestamp = datetime.now().isoformat()

    for domain_name in selected_domains:
        domain_dir = data_root / domain_name
        if not domain_dir.exists():
            print(f"[WARN] Skipping missing domain folder: {domain_dir}")
            continue

        problems_dir = domain_dir / "problems"
        observations_dir = domain_dir / "observations"

        if not problems_dir.exists():
            print(f"[WARN] No problems folder for {domain_name}, skipping")
            continue
        if not observations_dir.exists():
            print(f"[WARN] No observations folder for {domain_name}, skipping")
            continue

        problem_files = sorted(problems_dir.glob("*.pddl"))
        if not problem_files:
            print(f"[WARN] No problem files found under {problems_dir}")
            continue

        for idx, problem_path in enumerate(problem_files):
            problem_stem = problem_path.stem
            image_path = resolve_image_path(observations_dir, problem_stem)
            if image_path is None:
                print(f"[WARN] Missing observation for {problem_stem} ({observations_dir}), skipping entry")
                continue
            goal_instruction_path = resolve_goal_instruction_path(domain_dir, problem_stem)

            try:
                problem_entry = build_problem_entry(
                    domain_dir=domain_dir,
                    problem_path=problem_path,
                    image_path=image_path,
                    goal_instruction_path=goal_instruction_path,
                    dataset_timestamp=timestamp,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to process {problem_path}: {exc}")
                continue

            dataset["problems"].append(problem_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)
    print(f"[DONE] Wrote {len(dataset['problems'])} problems to {output_path}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDDLGym-style assets into an SSR grounding dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder containing domain subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional subset of domain folders to process. Defaults to every folder under data-root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination dataset path (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    generate_dataset(cli_args.data_root, cli_args.output_json, cli_args.domains)
