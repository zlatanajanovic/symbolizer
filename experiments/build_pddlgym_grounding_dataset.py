#!/usr/bin/env python3
"""
Build the PDDLGym grounding dataset for the paper's Table 1-3 domains.

The paper's PDDLGym grounding rows are **Blocksworld / Hanoi / Hanoi Color**.
Each sample is a single rendered PDDLGym state plus its ground-truth objects,
predicates and goal. The grounding evaluator (`experiments.run_grounding_eval`)
reads, per state:

    states[].image_path        rendered frame (the VLM input)
    states[].all_objects       GT objects   (JSON string)
    states[].atoms             GT predicates (JSON string)
    states[].goal_predicates   GT goal      (JSON string)
    states[].goal_instruction_path   NL goal instruction (txt)
    problem.domain_file        PDDL domain (object schema is derived from this)

This script assembles that dataset. It has two modes:

  --from-ssr <PHASE1_SSR_DIR>   Full (re)build. Reads the paper's saved SSR
                                states (domain.pddl / problem.pddl / state.json /
                                instruction.txt + GT objects/atoms/goal), copies
                                the source PDDL/text into the release tree,
                                re-renders every initial-state image with the
                                PDDLGym simulator renderer, and writes the
                                dataset JSON. Used once to construct the release.

  (default / --render-only)     Reviewer reproduction. Reads the already-bundled
                                dataset JSON and re-renders every image from the
                                bundled problem.pddl files. No external tree
                                needed -- the PDDL + GT are bundled; only the
                                rendered frames are regenerated.

Rendering is deterministic: each frame is the initial state of the bundled
problem.pddl, drawn by the same PDDLGym renderer used to produce the paper's
images (rollout_steps=0, image_mode=simulator).

Domains: dataset `domain_name` is the paper-pipeline value (blocks/hanoi/
hanoi_color); the Table 1-3 row label for `blocks` is "Blocksworld".
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

# Paper PDDLGym grounding domains -> the PDDLGym renderer that draws their states.
# (The `blocks` SSR domain is `blocks_operator_actions`; its literals use real
#  predicate names on/ontable/clear/holding, so the *plain* blocks renderer is
#  correct -- the "_encoded" renderer expects encoded predicate_N names.)
DOMAINS = ["blocks", "hanoi", "hanoi_color"]


def _get_renderer(domain_name):
    from pddlgym.rendering import blocks_render, hanoi_render, hanoi_color_render
    return {
        "blocks": blocks_render,
        "hanoi": hanoi_render,
        "hanoi_color": hanoi_color_render,
    }[domain_name]


def render_initial_state(domain_file, problem_file, renderer, out_path):
    """Render the initial state of `problem_file` to `out_path` (deterministic)."""
    from pddlgym.core import PDDLEnv

    # PDDLEnv globs <problem_dir>/*.pddl excluding files named '*domain*', so
    # isolate the single problem in a temp dir.
    pdir = tempfile.mkdtemp(prefix="pddlgym_render_")
    try:
        shutil.copy(problem_file, os.path.join(pdir, "problem.pddl"))
        env = PDDLEnv(
            domain_file, pdir, render=renderer,
            operators_as_actions=True, dynamic_action_space=True,
            raise_error_on_invalid_action=False,
        )
        env.fix_problem_index(0)
        state, _ = env.reset()
        arr = env._render(state.literals)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(arr).convert("RGB").save(out_path, quality=95)
        plt.close("all")  # the pddlgym renderers leave figures open
        return len(state.literals)
    finally:
        shutil.rmtree(pdir, ignore_errors=True)


def build_from_ssr(ssr_root, out_root, dataset_out):
    """Assemble the dataset + bundle source PDDL/text + render all images."""
    ds_dir = os.path.join(ssr_root, "datasets")
    out_problems = []
    n_rendered = 0

    for domain in DOMAINS:
        ssr_path = os.path.join(ds_dir, f"pddlgym_{domain}_ssr.json")
        if not os.path.exists(ssr_path):
            raise FileNotFoundError(f"SSR dataset not found: {ssr_path}")
        ssr = json.load(open(ssr_path))
        renderer = _get_renderer(domain)

        dom_dir = os.path.join(out_root, domain)
        prob_dir = os.path.join(dom_dir, "problems")
        obs_dir = os.path.join(dom_dir, "observations")
        instr_dir = os.path.join(dom_dir, "instructions")
        for d in (prob_dir, obs_dir, instr_dir):
            os.makedirs(d, exist_ok=True)

        domain_dst = os.path.join(dom_dir, "domain.pddl")
        domain_copied = False

        for prob in ssr["problems"]:
            name = prob["problem_name"]
            # SSR paths are repo-relative to the recovered tree root; each
            # sample's files live in the dir holding its state.json.
            sample_dir = os.path.dirname(
                os.path.join(_ssr_repo_root(ssr_root), prob["state_json_file"]))
            src_domain = os.path.join(sample_dir, "domain.pddl")
            src_problem = os.path.join(sample_dir, "problem.pddl")
            src_instr = os.path.join(sample_dir, "instruction.txt")

            if not domain_copied:
                shutil.copy(src_domain, domain_dst)
                domain_copied = True

            problem_dst = os.path.join(prob_dir, f"{name}.pddl")
            shutil.copy(src_problem, problem_dst)
            instr_dst = os.path.join(instr_dir, f"{name}.txt")
            if os.path.exists(src_instr):
                shutil.copy(src_instr, instr_dst)

            image_dst = os.path.join(obs_dir, f"{name}.jpg")
            n_lit = render_initial_state(domain_dst, problem_dst, renderer, image_dst)
            n_rendered += 1

            st0 = prob["states"][0]
            out_problems.append({
                "problem_name": name,
                "domain_name": domain,
                "domain_file": _rel(domain_dst),
                "problem_file": _rel(problem_dst),
                "states": [{
                    "id": st0.get("id", 0),
                    "image_path": _rel(image_dst),
                    "goal_instruction_path": _rel(instr_dst) if os.path.exists(instr_dst) else "",
                    "goal_predicates": st0["goal_predicates"],
                    "pickle_path_observation": "",
                    "all_objects": st0["all_objects"],
                    "atoms": st0["atoms"],
                    "atoms_schema": st0.get("atoms_schema", {}),
                    "actions": st0.get("actions", []),
                    "timestamp": st0.get("timestamp", ""),
                }],
            })
            print(f"  [{domain}] {name}: rendered ({n_lit} literals)")

    os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
    json.dump({"problems": out_problems}, open(dataset_out, "w"), indent=2)
    counts = {d: sum(1 for p in out_problems if p["domain_name"] == d) for d in DOMAINS}
    print(f"\nWrote {dataset_out}: {len(out_problems)} problems {counts}, "
          f"{n_rendered} images rendered.")


def render_only(dataset_path):
    """Re-render every image referenced by the bundled dataset (no external tree)."""
    ds = json.load(open(dataset_path))
    n = 0
    for prob in ds["problems"]:
        domain = prob["domain_name"]
        renderer = _get_renderer(domain)
        domain_file = os.path.join(REPO_ROOT, prob["domain_file"])
        problem_file = os.path.join(REPO_ROOT, prob["problem_file"])
        for st in prob["states"]:
            out_path = os.path.join(REPO_ROOT, st["image_path"])
            n_lit = render_initial_state(domain_file, problem_file, renderer, out_path)
            n += 1
            print(f"  [{domain}] {prob['problem_name']}: re-rendered ({n_lit} literals)")
    print(f"\nRe-rendered {n} images from {dataset_path}.")


def _ssr_repo_root(ssr_root):
    """The recovered repo root that SSR repo-relative paths are resolved against."""
    # SSR paths look like 'experiments/final_clean/phase1_ssr/data/...'; ssr_root
    # is '.../experiments/final_clean/phase1_ssr'. Strip 3 trailing components.
    return os.path.normpath(os.path.join(ssr_root, "..", "..", ".."))


def _rel(path):
    """Release-relative path (./experiments/...), forward slashes."""
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    return "./" + rel.replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-ssr", type=str, default=None,
                    help="Path to the recovered phase1_ssr dir for a full rebuild.")
    ap.add_argument("--render-only", action="store_true",
                    help="Only re-render images from the already-bundled dataset.")
    ap.add_argument("--out-root", type=str,
                    default=os.path.join(REPO_ROOT, "experiments/data/pddlgym_grounding"))
    ap.add_argument("--dataset-out", type=str,
                    default=os.path.join(REPO_ROOT, "experiments/data/datasets/pddlGYM_dataset_grounding.json"))
    args = ap.parse_args()

    if args.from_ssr:
        build_from_ssr(args.from_ssr, args.out_root, args.dataset_out)
    else:
        render_only(args.dataset_out)


if __name__ == "__main__":
    main()
