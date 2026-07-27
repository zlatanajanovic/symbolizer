#!/usr/bin/env bash
# Self-contained offline smoke test for the SYMBOLIZER release.
# Builds the Docker image and verifies imports, CLIs, data wiring, the offline
# symbolic-planning reproduction, and PDDLGym rendering (the visualization path)
# — WITHOUT making any API calls.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-symbolizer-release:latest}"
SMOKE_STAGE_DELAY="${SMOKE_STAGE_DELAY:-15}"

docker build -t "$IMAGE" "$ROOT_DIR"

run_in_image() {
  local name="$1"
  local cmd="$2"
  echo
  echo "==> $name"
  # Give Docker/cgroup memory accounting a moment to settle between the
  # import-heavy Python stages. On constrained hosts, launching the next
  # container immediately after the previous one exits can produce exit 137.
  sleep "$SMOKE_STAGE_DELAY"
  docker run --rm --init \
    -e MALLOC_ARENA_MAX=2 \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    "$IMAGE" bash -lc "
set -euo pipefail
cd /workspace
$cmd
"
}

run_in_image "import checks" '
python - <<PY
import symbolizer
import symbolizer.ssr_parsing
import pddlgym
from pddlgym.inference import check_goal
from symbolizer.search_eval.simulator.viplan_blocksworld_simulator import VIPLAN_AVAILABLE
print("import checks OK (incl. viplan simulator; live render+VLM demo: examples/viplan_blocksworld_vlm_example.py)")
PY
'

# Data wiring and CLI entrypoint: enumerate the full grounding work-list from
# bundled data (no API).
run_in_image "grounding dry-run (bundled data)" '
python -m experiments.run_grounding_eval --config config.yaml --dataset all \
  --dry-run --output_dir /tmp/smoke-grounding >/dev/null && echo "grounding dry-run (bundled data) OK"
'

# Planning CLI entrypoint: offline symbolic planning over the checked-in problems.
run_in_image "planning dry-run" '
python -m experiments.run_planning_benchmark --dry_run \
  --data_dir experiments/data/planning --output_dir /tmp/smoke-planning >/dev/null && echo "planning dry-run OK"
'

# Offline planning reproduction: re-execute the shipped symbolic planner on the
# bundled problems (reproduces the offline planning-success tables, no API).
# Exits non-zero only on a real planner failure.
run_in_image "offline planning reproduction (planner re-exec)" '
python -m experiments.regenerate_all_tables --smoke >/tmp/smoke-tables.log 2>&1 \
  && echo "offline planning reproduction OK" \
  || { echo "regenerate_all_tables FAILED"; tail -20 /tmp/smoke-tables.log; exit 1; }
'

# Rendering / visualization: render the initial state of every PDDLGym grounding
# domain (Blocksworld / Hanoi / Hanoi Color) headlessly and assert valid frames.
# This is the visualization path the grounding dataset and the web demo rely on.
run_in_image "rendering (visualization)" '
python - <<PY
import os, sys, json, tempfile
os.environ.setdefault("MPLBACKEND", "Agg")
from experiments.build_pddlgym_grounding_dataset import render_initial_state, _get_renderer
from PIL import Image
ds = json.load(open("experiments/data/datasets/pddlGYM_dataset_grounding.json"))
seen = {}
for p in ds["problems"]:
    seen.setdefault(p["domain_name"], p)
assert set(seen) == {"blocks", "hanoi", "hanoi_color"}, sorted(seen)
for d, p in seen.items():
    out = os.path.join(tempfile.mkdtemp(), d + ".png")
    n = render_initial_state(p["domain_file"], p["problem_file"], _get_renderer(d), out)
    w, h = Image.open(out).size
    assert w > 10 and h > 10 and os.path.getsize(out) > 1000, (d, w, h)
    print(f"  rendered {d}: {w}x{h}px ({n} literals)")
print("rendering (visualization) OK")
PY
'

echo "SMOKE TEST PASSED"
