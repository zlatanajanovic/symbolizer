import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

SENSITIVE_KEYS = ("api_key", "apikey", "secret", "token", "password")
# Example usage:
#   python -m symbolizer.ssr_parsing.run_full_pipeline 
from .parsing_utils.general_utils import load_config
from .run_eval_vlm import (
    calculate_metrics,
    calculate_metrics_objects,
    calculate_metrics_per_domain,
    calculate_metrics_objects_per_domain,
)


def _load_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict]:
    results: List[Dict] = []
    if not path.exists():
        return results

    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _resolve_result_file(value: str, output_folder: Path, repo_base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("./") or value.startswith("../"):
        return (repo_base / path).resolve()
    return (output_folder / path).resolve()


def _run_command(cmd: List[str], cwd: Path) -> None:
    print(f"\n[run_full_pipeline] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _gather_domain_metadata(dataset_path: Path) -> Dict[str, Dict[str, object]]:
    dataset = _load_json(dataset_path)
    problems = dataset.get("problems", [])

    domain_map: Dict[str, Dict[str, object]] = {}
    for entry in problems:
        domain_name = entry.get("domain_name")
        domain_file = entry.get("domain_file")
        problem_file = entry.get("problem_file")

        if not domain_name or not domain_file or not problem_file:
            continue

        info = domain_map.setdefault(
            domain_name,
            {
                "data_dir": Path(domain_file).resolve().parent,
                "problem_names": set(),
            },
        )
        info["problem_names"].add(Path(problem_file).stem)

    for domain_name, info in domain_map.items():
        info["num_gt_problems"] = len(info["problem_names"])
    return domain_map


def _compute_generated_repeat(result_dir: Path, num_gt: int) -> int:
    problems_dir = result_dir / "problems"
    if not problems_dir.exists():
        return 0
    generated = len(list(problems_dir.glob("*.pddl")))
    if num_gt <= 0 or generated <= 0:
        return 0
    if generated % num_gt == 0:
        return generated // num_gt
    # Fallback: round to the nearest integer >= 1
    return max(1, round(generated / num_gt))


def _collect_metrics(summary: Dict[str, object], config: Dict[str, object], repo_base: Path) -> None:
    output_folder = _resolve_path(config["output_folder"], repo_base)
    objects_path = _resolve_result_file(config["objects_full_responses_path"], output_folder, repo_base)
    atoms_path = _resolve_result_file(config["atoms_full_responses_path"], output_folder, repo_base)
    goals_path = _resolve_result_file(config["goals_full_responses_path"], output_folder, repo_base)

    evaluation_results_objects = _load_jsonl(objects_path)
    evaluation_results_atoms = _load_jsonl(atoms_path)
    evaluation_results_goals = _load_jsonl(goals_path)

    metrics = {
        "objects": {
            "overall": calculate_metrics_objects(evaluation_results_objects),
            "per_domain": calculate_metrics_objects_per_domain(evaluation_results_objects),
            "count": len(evaluation_results_objects),
        },
        "atoms": {
            "overall": calculate_metrics(evaluation_results_atoms),
            "per_domain": calculate_metrics_per_domain(evaluation_results_atoms),
            "count": len(evaluation_results_atoms),
        },
        "goals": {
            "overall": calculate_metrics(evaluation_results_goals),
            "per_domain": calculate_metrics_per_domain(evaluation_results_goals),
            "count": len(evaluation_results_goals),
        },
    }

    summary["metrics"] = metrics


def _evaluate_with_vilain(
    summary: Dict[str, object],
    dataset_path: Path,
    output_folder: Path,
    downward_dir: Path,
    gen_step: str,
    repo_base: Path,
) -> None:
    domain_map = _gather_domain_metadata(dataset_path)
    results_dir = output_folder / "problems"

    evaluation_records: List[Dict[str, object]] = []
    for domain, info in domain_map.items():
        domain_result_dir = results_dir / domain
        problems_dir = domain_result_dir / "problems"

        if not problems_dir.exists():
            print(f"[run_full_pipeline] Skipping ViLaIn eval for {domain}: generated problems not found.")
            continue

        num_gt = info.get("num_gt_problems", 0)
        num_repeat = _compute_generated_repeat(domain_result_dir, num_gt)
        if num_repeat == 0:
            print(f"[run_full_pipeline] Skipping {domain}: unable to infer num_repeat (gt={num_gt}).")
            continue

        cmd = [
            sys.executable,
            "experiments/ViLaIn/scripts/evaluate.py",
            "--downward_dir",
            str(downward_dir),
            "--data_dir",
            str(info["data_dir"]),
            "--result_dir",
            str(domain_result_dir),
            "--num_repeat",
            str(num_repeat),
            "--gen_step",
            gen_step,
        ]
        _run_command(cmd, cwd=repo_base)

        evaluation_records.append(
            {
                "domain": domain,
                "result_dir": str(domain_result_dir),
                "data_dir": str(info["data_dir"]),
                "num_gt_problems": num_gt,
                "num_repeat": num_repeat,
            }
        )

    summary["vilain_evaluations"] = evaluation_records


def _write_sanitized_env_copy(env_path: Path, output_folder: Path) -> Optional[Path]:
    if not env_path.exists():
        print(f"[run_full_pipeline] Skipping env snapshot: {env_path} not found.")
        return None

    sanitized_lines: List[str] = []
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        if any(token in key.lower() for token in SENSITIVE_KEYS):
            continue

        sanitized_lines.append(f"{key.strip()}={value.strip()}")

    if not sanitized_lines:
        print("[run_full_pipeline] Env snapshot skipped: no non-sensitive fields found.")
        return None

    snapshot_path = output_folder / "env_snapshot.txt"
    snapshot_path.write_text("\n".join(sanitized_lines) + "\n")
    print(f"[run_full_pipeline] Env snapshot written to {snapshot_path}")
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: generation -> metrics -> PDDL -> ViLaIn evaluation.")
    parser.add_argument("--config", default="config.yaml", help="Path to generation config (default: config.yaml).")
    parser.add_argument("--config-eval", default="config_eval.yaml", help="Path to post-processing config (default: config_eval.yaml).")
    parser.add_argument("--downward-dir", default="./external/downward", help="Path to Fast-Downward directory (required for ViLaIn evaluation).")
    parser.add_argument("--gen-step", default="plain", help="Generation step label for ViLaIn evaluation (default: plain).")
    parser.add_argument("--skip-generation", action="store_true", help="Skip the run_eval_on_dataset step.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metric aggregation.")
    parser.add_argument("--skip-pddl", action="store_true", help="Skip conversion of JSONL outputs into PDDL problems.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the ViLaIn comparison step.")
    args = parser.parse_args()

    repo_base = _repo_root()
    generation_config = load_config(args.config)
    eval_config = load_config(args.config_eval)

    output_folder = _resolve_path(generation_config["output_folder"], repo_base)
    output_folder.mkdir(parents=True, exist_ok=True)

    env_snapshot_path: Optional[Path] = None
    env_file = generation_config.get("env_file")
    if env_file:
        env_snapshot_path = _write_sanitized_env_copy(_resolve_path(env_file, repo_base), output_folder)

    summary: Dict[str, object] = {
        "config": _resolve_path(args.config, repo_base).as_posix(),
        "config_eval": _resolve_path(args.config_eval, repo_base).as_posix(),
        "output_folder": output_folder.as_posix(),
        "steps": [],
    }
    if env_snapshot_path:
        summary["env_snapshot"] = env_snapshot_path.as_posix()

    if not args.skip_generation:
        _run_command(
            [sys.executable, "-m", "symbolizer.ssr_parsing.run_eval_on_dataset"],
            cwd=repo_base,
        )
        summary["steps"].append("generation")

    if not args.skip_pddl:
        _run_command(
            [sys.executable, "symbolizer/symbolizer/ssr_parsing/pddl_from_json.py"],
            cwd=repo_base,
        )
        summary["steps"].append("pddl_conversion")

    if not args.skip_metrics:
        try:
            _collect_metrics(summary, eval_config, repo_base)
            summary["steps"].append("metrics")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[run_full_pipeline] Failed to collect metrics: {exc}")

    if not args.skip_evaluation:
        downward_dir = args.downward_dir or eval_config.get("downward_dir")
        if not downward_dir:
            raise ValueError("downward_dir is required for ViLaIn evaluation (pass --downward-dir or set it in config_eval.yaml).")

        _evaluate_with_vilain(
            summary,
            _resolve_path(eval_config["dataset_path"], repo_base),
            output_folder,
            _resolve_path(downward_dir, repo_base),
            args.gen_step,
            repo_base,
        )
        summary["steps"].append("vilain_evaluation")

    summary_path = output_folder / "pipeline_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n[run_full_pipeline] Pipeline completed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
