# Simulators & Experiments — what runs, and how

One command runs a smoke of **every** simulator and reports PASS / SKIP / FAIL:

```bash
# Offline only (no API, no ViPlan):
bash run_all_simulators.sh

# + live VLM steps (grounding, VLM-grounded planning):
GAC=~/.config/gcp/vertex.json bash run_all_simulators.sh

# + ViPlan (blocksworld render + VLM grounding):
GAC=~/.config/gcp/vertex.json VIPLAN_ROOT=~/ViPlan bash run_all_simulators.sh
```

`GAC` = path to a Vertex service-account JSON. `VIPLAN_ROOT` = a ViPlan checkout
that has been set up with Blender (see ViPlan section). The script auto-selects
`symbolizer-viplan:latest` if built, else `symbolizer-release:latest`.

## What runs — and what each path needs

A ✅ marks a path that is supported out of the box once the prerequisites in the
**Needs** column are met (offline paths need nothing extra). Use the command in
the **How** column to run each one; `run_all_simulators.sh` chains them and prints
`PASS` / `SKIP` / `FAIL` for your own run.

| Simulator / experiment | Supported | How | Needs |
|---|---|---|---|
| **PDDLGym** grounding (obj/pred/goal) | ✅ | `run_grounding_eval --dataset pddlgym` | VLM creds |
| **PDDLGym** simulator + rendering | ✅ | `pddlgym.make(...).render()` | — (offline) |
| **PDDLGym/PyBullet/Kitchen** symbolic planning | ✅ | `run_planning_benchmark` | — (offline) |
| **ViLaIn (ProDG)** grounding | ✅ | `run_grounding_eval --dataset vilain` | VLM creds |
| **Real-world image** grounding (objects/predicates) | ✅ | `run_grounding_eval --dataset real` | VLM creds |
| **VLM-grounded A\*** (VLM in the loop) | ✅ | `run_search_comparison --env PDDLEnvHanoi-v0` | VLM creds, ≥ a few GB RAM |
| End-to-end example (image→ground→plan) | ✅ | `examples/run_example.py` | VLM creds |
| **ViPlan Blocksworld** symbolic planning | ✅ | `run_planning_benchmark` w/ `VIPLAN_ROOT` | ViPlan checkout |
| **ViPlan Blocksworld** Blender render | ✅ | `BlocksworldSimulator(...).render()` | ViPlan + Blender |
| **ViPlan Blocksworld** VLM grounding | ✅ | `examples/viplan_blocksworld_vlm_example.py` | ViPlan + Blender + VLM creds, ≥ a few GB RAM |
| Reproduce planning Tables 4–5 (offline) | ✅ | `regenerate_all_tables.py --smoke` | — (offline) |
| Reproduce grounding Tables 1–3 (live) | ✅ | `run_grounding_eval --dataset {pddlgym,real,vilain}` | VLM creds |

Symbolic ViPlan-Household planning **does** run (it uses the bundled symbolic
data via the `pddlgym` backend in `run_planning_benchmark`).

**Goal-grounding caveat.** Object and predicate grounding run live (with VLM
creds) for the grounding rows above. *Goal* grounding additionally needs each problem's natural-language
goal-instruction file: these are bundled for the **PDDLGym** and **ViLaIn
(`--dataset vilain`)** sets, so their goal F1 is fully reproducible. They were
**not** released for the **real-world image** set, so a live `--dataset real`
run cannot regenerate real goal F1 — the Table 3 "Real Images" goal cell is the
value reported in the paper.

## Docker images

```bash
docker build -t symbolizer-release:latest .                 # base: grounding + symbolic + VLM-grounded planning
docker build -f Dockerfile.viplan -t symbolizer-viplan:latest .   # + ViPlan deps + headless Blender libs
```

`symbolizer-viplan` adds `torch` (CPU), `mloggers`, `fasteners`, `matplotlib`, and
the X/GL system libraries Blender needs for headless CYCLES rendering.

## ViPlan setup (Blocksworld)

```bash
git clone https://github.com/merlerm/ViPlan.git ~/ViPlan
(cd ~/ViPlan && bash setup_blocksworld.sh)     # downloads Blender 3.0 into the checkout
export VIPLAN_ROOT=~/ViPlan
```

Rendering defaults to **CPU** (`VIPLAN_USE_GPU_RENDERING=0`, portable). On a CUDA
host set `VIPLAN_USE_GPU_RENDERING=1` for faster Blender renders.

## PDDLGym grounding dataset = the paper's Table 1–3 domains

`experiments/data/datasets/pddlGYM_dataset_grounding.json` contains exactly the
paper's Table 1–3 **PDDLGym** domains and nothing else:

| `domain_name` | Table 1–3 row | samples |
|---|---|---|
| `blocks`       | Blocksworld | 25 |
| `hanoi`        | Hanoi       | 25 |
| `hanoi_color`  | Hanoi Color | 25 |

Each sample bundles its source PDDL (`pddlgym_grounding/<domain>/{domain.pddl,
problems/*.pddl}`), goal instruction (`instructions/*.txt`), the rendered initial
state (`observations/*.jpg`), and ground-truth objects/predicates/goal (inside the
dataset JSON). Live grounding F1 matches the paper's reported values, e.g.
Blocksworld objects/predicates/goal ≈ 0.98/0.98/0.96 for Gemini 3.1 Flash-Lite
(1.0/1.0/1.0 for Gemini 3.1 Pro) and Hanoi predicates ≈ 0.90.

### Regenerating the rendered frames

The images are produced deterministically from the bundled `problem.pddl` files by
the PDDLGym simulator renderer (initial state, `rollout_steps=0`):

```bash
python -m experiments.build_pddlgym_grounding_dataset            # re-render from bundled PDDL
python -m experiments.build_pddlgym_grounding_dataset --from-ssr <phase1_ssr_dir>  # full rebuild
```

`--render-only`/default needs no external data — it re-renders every frame from
the bundled problems. The full `--from-ssr` rebuild (used to construct the
release) additionally copies the source PDDL/text and ground truth from the
paper's saved SSR states. (Earlier release snapshots shipped the wrong
*prediction-experiment* dataset here — next-state-prediction domains, not the
paper's grounding domains — which has been removed entirely.)

## Notes

- Live VLM steps work with Gemini (Vertex), OpenAI, or any OpenAI-compatible
  endpoint (`OPENAI_BASE_URL`); set the provider in `.env`. Do not re-run Mistral
  casually (cost).
- The VLM-in-the-loop steps (VLM-grounded A\*, ViPlan VLM grounding) render
  high-res frames and call the VLM repeatedly — give them several GB of free RAM.
