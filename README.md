# SYMBOLIZER: Symbolic Model-free Task Planning with VLMs

Code for the paper *"Symbolizer: Symbolic Model-free Task Planning with VLMs"*.

SYMBOLIZER is a VLM-based grounding module that extracts symbolic representations (objects, predicates, goals) from images and natural language using **structured JSON output**. Combined with classical search algorithms (A\* with goal-count heuristic, width-based search), it enables end-to-end visual planning.

## Repository Structure

```
symbolizer/
├── symbolizer/               # Core library
│   ├── ssr_parsing/          # VLM grounding: objects, predicates, goals
│   │   ├── ssr_parsing.py    # Core: retrieve_objects, retrieve_predicates, retrieve_goal
│   │   ├── run_eval_vlm.py   # VLM client + evaluation logic
│   │   ├── pddl_from_json.py # JSON -> PDDL conversion
│   │   └── parsing_utils/    # Prompt generation, schema construction
│   └── search_eval/          # Search algorithms + simulators
│       ├── search/           # IWK, SIW, BFWS, A*, BFS, novelty
│       └── simulator/        # PDDLGym, PDDL file, ViPlan simulators
├── pddlgym/                  # Modified PDDLGym (blocks + hanoi rendering)
├── examples/                 # End-to-end examples (image→ground→plan; ViPlan render + VLM grounding)
├── experiments/              # Scripts to reproduce paper tables
│   ├── run_grounding_eval.py       # Tables 1-3: object/predicate/goal F1
│   ├── run_planning_benchmark.py   # Tables 4-6: planning success (incl. ViPlan-Blocks symbolic)
│   └── data/                       # Domain PDDL files, sample images, datasets
└── experiments/results/       # (runtime outputs only; not shipped — this is a code-only release)
```

## Setup

### 1. Install the package

```bash
python -m pip install -e ./pddlgym
python -m pip install -e .
```

The editable installs make both the modified bundled PDDLGym package and
`symbolizer` importable without setting `PYTHONPATH`.

### 2. Configure API keys

```bash
cp .env_template .env
# Edit .env with your API keys (OpenAI, Gemini, Mistral, etc.)
```

### 3. Verify installation

```bash
python -c "from symbolizer.ssr_parsing.ssr_parsing import retrieve_objects; print('OK')"
```

### 4. Build & verify with Docker (self-contained)

```bash
# Builds the image and runs offline checks (imports, CLIs, data wiring,
# planning dry-run, regeneration tooling + planner re-execution) — no API calls:
bash smoke_test.sh

# Or keep a container up for interactive use:
docker compose up -d --build symbolizer
docker exec -it symbolizer_release bash
```

The repository is self-contained: the Dockerfile, `docker-compose.yml`, and
`smoke_test.sh` are included, and all grounding datasets ship with their images
under `experiments/data/` so the dry-run enumerates every problem out of the box.

### Run every simulator

One command runs a smoke of **every** simulator/experiment and reports
PASS/SKIP/FAIL. See [`SIMULATORS.md`](SIMULATORS.md) for the full matrix and setup.

```bash
bash run_all_simulators.sh                                      # offline only
GAC=~/.config/gcp/vertex.json bash run_all_simulators.sh        # + live grounding/planning
GAC=~/.config/gcp/vertex.json VIPLAN_ROOT=~/ViPlan bash run_all_simulators.sh   # + ViPlan
```

The offline steps (symbolic planning over PDDLGym/PyBullet/Kitchen, planner
re-execution) need no credentials and run from the bundled data. The live steps
(ViLaIn/PDDLGym/real grounding, VLM-grounded A\*) run when you pass
`GAC=path/to/vertex.json`; the ViPlan Blocksworld render+grounding step runs when
you also pass `VIPLAN_ROOT`. Each step prints `PASS` / `SKIP` / `FAIL` for your
own run, so you can see exactly what executed on your machine.

## Quick Start

### End-to-end example

```bash
# Live: ground a bundled image (objects/predicates/goal) and plan with A*.
# Requires a configured provider (.env or env vars); makes real VLM calls.
python examples/run_example.py                                   # pddlgym/hanoi
python examples/run_example.py --dataset vilain --domain_name blocksworld_with_robot
python examples/run_example.py --dataset real --domain_name blocksworld_real
```

### Web Demo

An interactive demo is available at: **<https://symbolizer-symbolizer.hf.space>**
(also reachable from the [HuggingFace Space page](https://huggingface.co/spaces/Symbolizer/SYMBOLIZER)).

**Access password:** `[REDACTED_FOR_PUBLIC_RELEASE]`

> If the password screen reappears right after you submit the password, your
> browser is blocking the login cookie inside the embedded Space preview. Open
> the demo at its direct URL above (a normal full browser tab) and log in there.

Log in with the password, then upload a Blocks World or Tower of Hanoi image, and the
system will ground it symbolically, visualize the state, and plan a solution with A\*
search. A grounding model is built in, so no API key is required; you can also supply
your own Gemini / OpenAI / Vertex AI key in the UI.

## Reproducing Paper Results

This is a **code-only** release: it ships the experiment code and all the input
data needed to reproduce every table, but **not** our precomputed result files.
You reproduce the numbers by running the experiments yourself and comparing
against the values reported in the paper.

### Offline planning tables — one command, no API, no cost

```bash
# Re-execute the shipped symbolic planner on the bundled problems. This
# reproduces the offline planning-success numbers (Tables 4-5 and the symbolic
# ViPlan rows of Table 6): solved / total per domain for A* and A*+Novelty.
python -m experiments.regenerate_all_tables --smoke

# Print the live, from-scratch re-run recipe for every table:
python -m experiments.regenerate_all_tables --print-live-commands
```

The grounding tables (Tables 1-3) and the VLM-in-the-loop rows make live VLM
calls — see the per-table commands below and in
[`experiments/REGENERATE_TABLES.md`](experiments/REGENERATE_TABLES.md). A fresh
live re-run is a new sample and may differ slightly from the paper's printed
values due to VLM nondeterminism.

### Tables 1-3: Grounding F1 (Object / Predicate / Goal)

```bash
# PDDLGym grounding evaluation (Blocksworld / Hanoi / Hanoi Color)
python -m experiments.run_grounding_eval --config config.yaml --dataset pddlgym

# Real image grounding evaluation
python -m experiments.run_grounding_eval --config config.yaml --dataset real

# ViLaIn grounding evaluation
python -m experiments.run_grounding_eval --config config.yaml --dataset vilain
```

The PDDLGym grounding dataset (`experiments/data/datasets/pddlGYM_dataset_grounding.json`)
holds exactly the paper's Table 1-3 domains — `blocks` (Blocksworld), `hanoi`,
`hanoi_color` — 25 samples each, with ground-truth objects/predicates/goal and the
rendered initial-state frames under `experiments/data/pddlgym_grounding/`. The frames
are reproduced deterministically from the bundled `problem.pddl` files with the PDDLGym
simulator renderer:

```bash
python -m experiments.build_pddlgym_grounding_dataset   # re-render all frames from bundled PDDL
```

Live F1 matches the paper raw (e.g. Blocksworld object/predicate/goal ≈ 0.98/0.98/0.96;
Hanoi predicates ≈ 0.90, Gemini 3.1 Flash-Lite; 1.0/1.0/1.0 for Gemini 3.1 Pro).

### Tables 4-5: Planning Success Rate

```bash
# Symbolic search comparison across all domains
python -m experiments.run_planning_benchmark \
    --data_dir experiments/data/planning \
    --output_dir experiments/results/generated/planning

# With custom budget
python -m experiments.run_planning_benchmark \
    --budget 10000 \
    --output_dir experiments/results/generated/planning_budget_10000

# No-API dry run that only enumerates benchmark inputs
python -m experiments.run_planning_benchmark --dry_run
```

`run_planning_benchmark` is offline symbolic search (no VLM). The `ViPlan-Blocks`
domain is skipped unless a ViPlan checkout is available (see below); all other
domains run from the bundled data.

### VLM-grounded planning (VLM in the loop)

`run_search_comparison.py` runs the full Symbolizer planning loop: the VLM grounds
each rendered PDDLGym state during search (objects + predicates), and A* searches
the true transitions guided by the VLM goal-count heuristic. This makes live VLM
calls and renders high-res frames — run it on a host with several GB of free RAM.

```bash
python -m experiments.run_search_comparison \
    --env PDDLEnvHanoi-v0 --problem_index 0 \
    --model gemini-3.1-flash-lite-preview --max_vlm_obs 15 --max_astar_exp 60
# -> grounds each state with the VLM and returns an A* plan (e.g. 7-step Hanoi solution)
```

### Table 6: ViPlan Benchmark

Table 6 uses the external [ViPlan benchmark](https://github.com/merlerm/ViPlan).
Clone it, set `VIPLAN_ROOT`, and use the ViPlan-extended image (adds torch /
mloggers / fasteners / matplotlib; Blender is only needed to *render* images, not
for the symbolic simulator):

```bash
git clone https://github.com/merlerm/ViPlan.git /path/to/ViPlan
(cd /path/to/ViPlan && bash setup_blocksworld.sh)   # downloads Blender 3.0 (render only)
docker build -t symbolizer-release:latest .
docker build -f Dockerfile.viplan -t symbolizer-viplan:latest .

# Symbolic ViPlan planning — the ViPlan-Blocks simulator now runs in the benchmark
# (no longer skipped); ViPlan-Household runs from the bundled symbolic data:
docker run --rm -e VIPLAN_ROOT=/viplan -v /path/to/ViPlan:/viplan \
    -v "$PWD":/workspace -w /workspace symbolizer-viplan:latest \
    python -m experiments.run_planning_benchmark --data_dir experiments/data/planning

# Live ViPlan Blocksworld render (Blender) + VLM grounding on the rendered image
# (add your VLM provider env, e.g. Vertex creds — see "Models" below):
docker run --rm -e VIPLAN_ROOT=/viplan -v /path/to/ViPlan:/viplan \
    -v "$PWD":/workspace -w /workspace symbolizer-viplan:latest \
    python examples/viplan_blocksworld_vlm_example.py
```

The two commands above are the live symbolic and render+grounding paths for
Table 6; compare the success rates they report against the paper's Table 6.

## Models

The paper evaluates the following VLMs:
- **Gemini 3.1 Pro** (`gemini-3.1-pro`)
- **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite`)
- **Mistral Small 2503** (`mistral-small-2503`)

Set the model in `.env`:
```
MODEL_TO_USE=gemini
MODEL_NAME=gemini-3.1-pro
```

### Other providers (OpenAI, vLLM, OpenAI-compatible servers)

`MODEL_TO_USE` accepts `gemini`, `openai`, `mistral`, or `selfhosted`. Any
OpenAI-compatible endpoint (a local **vLLM** server, LiteLLM, OpenRouter, ...)
works via the `openai` provider plus a custom base URL:

```
MODEL_TO_USE=openai
OPENAI_BASE_URL=http://localhost:8000/v1     # your vLLM / proxy endpoint
OPENAI_API_KEY=EMPTY                          # placeholder if the server ignores it
MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
```

For Gemini via Vertex AI, set `GEMINI_USE_VERTEXAI=true`, `PROJECT`, `LOCATION`,
and `GOOGLE_APPLICATION_CREDENTIALS` (path to a service-account JSON).

## Domains

| Domain | Source | Type |
|--------|--------|------|
| Blocksworld | PDDLGym | Symbolic + rendered |
| Hanoi | PDDLGym | Symbolic + rendered |
| Hanoi Color | PDDLGym | Symbolic + rendered |
| Blocksworld | PyBullet | Physics sim |
| Hanoi | PyBullet | Physics sim |
| Kitchen-Worlds | Custom | Household tasks |
| Blocksworld | ViPlan | Photorealistic (Blender) |
| Household | ViPlan | Symbolic |
