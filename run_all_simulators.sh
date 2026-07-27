#!/usr/bin/env bash
# Run a smoke of EVERY simulator / experiment and report PASS / SKIP / FAIL.
#
# Offline steps always run. Live VLM steps run only when credentials are given;
# ViPlan steps run only when a ViPlan checkout is given. Nothing is fatal — each
# step is reported on its own so you see exactly what runs.
#
# Optional env knobs:
#   IMAGE        docker image (default: symbolizer-viplan:latest, else symbolizer-release:latest)
#   GAC          host path to a Vertex service-account JSON  -> enables live VLM steps
#   VIPLAN_ROOT  host path to a ViPlan checkout (with Blender) -> enables ViPlan steps
#   MODEL_NAME   VLM model (default: gemini-3.1-flash-lite-preview)
#   PROJECT/LOCATION             Vertex project/location (project auto-read from GAC)
#   VIPLAN_USE_GPU_RENDERING     0=CPU (default), 1=GPU
#
# Examples:
#   bash run_all_simulators.sh
#   GAC=~/.config/gcp/vertex.json bash run_all_simulators.sh
#   GAC=~/.config/gcp/vertex.json VIPLAN_ROOT=~/ViPlan bash run_all_simulators.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_STAGE_DELAY="${SIM_STAGE_DELAY:-15}"

# ---- configuration ---------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-gemini-3.1-flash-lite-preview}"
LOCATION="${LOCATION:-global}"
GAC="${GAC:-}"
VIPLAN_ROOT="${VIPLAN_ROOT:-}"
PROJECT="${PROJECT:-}"

HAVE_VLM=false;    [ -n "$GAC" ] && [ -f "$GAC" ] && HAVE_VLM=true
HAVE_VIPLAN=false; [ -n "$VIPLAN_ROOT" ] && [ -d "$VIPLAN_ROOT" ] && HAVE_VIPLAN=true

# Use the heavier ViPlan image (bundles torch + Blender libs) only when ViPlan
# steps are requested; otherwise the lighter base image (lower memory).
IMAGE="${IMAGE:-}"
if [ -z "$IMAGE" ]; then
  if $HAVE_VIPLAN && docker image inspect symbolizer-viplan:latest >/dev/null 2>&1; then
    IMAGE=symbolizer-viplan:latest
  else
    IMAGE=symbolizer-release:latest
  fi
fi
if $HAVE_VLM && [ -z "$PROJECT" ]; then
  PROJECT="$(python3 -c "import json;print(json.load(open('$GAC')).get('project_id',''))" 2>/dev/null || true)"
fi

# docker args for live Vertex VLM steps / for ViPlan steps (bash arrays)
VLM_ARGS=(); $HAVE_VLM && VLM_ARGS=(
  -e MODEL_TO_USE=gemini -e MODEL_NAME="$MODEL_NAME" -e GEMINI_USE_VERTEXAI=true
  -e PROJECT="$PROJECT" -e LOCATION="$LOCATION"
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds/vertex.json -v "$GAC":/creds/vertex.json:ro
)
VIPLAN_ARGS=(); $HAVE_VIPLAN && VIPLAN_ARGS=(
  -e VIPLAN_ROOT=/viplan -e VIPLAN_USE_GPU_RENDERING="${VIPLAN_USE_GPU_RENDERING:-0}"
  -e PYTHONPATH=/viplan -v "$VIPLAN_ROOT":/viplan
)

# ---- helpers ---------------------------------------------------------------
PASS=(); SKIP=(); FAIL=()
NOISE='Gym has been|Please upgrade|Users of this|migration guide|Encode image|Image Path|^True$|^512$|len messages|gemini_client|Auto-init|Using Gemini|Sending API|Response from Model|httpx|google_genai|AFC is'

# run_step NAME EXPECT CMD  [docker args via the DARGS array]
run_step() {
  local name="$1" expect="$2" cmd="$3" tmp rc
  echo; echo "############################################################"
  echo "# $name"; echo "############################################################"
  tmp="$(mktemp)"
  sleep "$SIM_STAGE_DELAY"
  timeout 600 docker run --rm --init \
    -e MALLOC_ARENA_MAX=2 \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    "${DARGS[@]}" \
    -v "$ROOT":/workspace -w /workspace "$IMAGE" bash -lc "$cmd" >"$tmp" 2>&1
  rc=$?
  grep -vE "$NOISE" "$tmp" | tail -40
  if grep -qF "$expect" "$tmp"; then
    PASS+=("$name"); echo ">> PASS: $name"
  else
    # rc 137 = OOM-killed, 124 = timeout — usually host resource limits, not a code bug.
    FAIL+=("$name (docker exit $rc)"); echo ">> FAIL: $name (docker exit $rc)"
  fi
  rm -f "$tmp"
}

echo "IMAGE=$IMAGE  MODEL=$MODEL_NAME"
echo "live VLM steps: $($HAVE_VLM && echo ENABLED || echo 'SKIPPED (set GAC=path/to/vertex.json)')"
echo "ViPlan steps:   $($HAVE_VIPLAN && echo ENABLED || echo 'SKIPPED (set VIPLAN_ROOT=path/to/ViPlan)')"

# ---- 1. offline symbolic planning (PDDLGym/PyBullet/Kitchen [+ ViPlan]) -----
DARGS=("${VIPLAN_ARGS[@]}")
run_step "Symbolic planning (PDDLGym/PyBullet/Kitchen[/ViPlan])" "TOTAL (all probs)" \
  "python -m experiments.run_planning_benchmark --data_dir experiments/data/planning --budget 3000 --output_dir /tmp/p"

# ---- 2. offline: planning reproduction (shipped symbolic planner re-exec) ---
DARGS=()
run_step "Offline planning reproduction (planner re-exec)" "RESULT:" \
  "python -m experiments.regenerate_all_tables --smoke"

# ---- 3. live grounding (Tables 1-3) on each dataset -------------------------
if $HAVE_VLM; then
  DARGS=("${VLM_ARGS[@]}")
  for d in vilain pddlgym real; do
    # The pddlgym grounding dataset holds exactly the paper's Table 1-3 domains
    # (Blocksworld/Hanoi/Hanoi Color); any sampled domain is representative.
    run_step "Grounding ($d)" "Goals Evaluation Metrics" \
      "python -m experiments.run_grounding_eval --config config.yaml --dataset $d --num_tests 1 --output_dir /tmp/g_$d"
  done
  # ---- 4. live VLM-grounded A* planning (VLM in the loop) --------------------
  run_step "VLM-grounded A* planning (hanoi)" "VLM-grounded A* result" \
    "python -m experiments.run_search_comparison --env PDDLEnvHanoi-v0 --problem_index 0 --max_vlm_obs 12 --max_astar_exp 40"
else
  SKIP+=("Grounding (vilain/pddlgym/real) — set GAC" "VLM-grounded A* planning — set GAC")
fi

# ---- 5. ViPlan blocksworld render + VLM grounding --------------------------
if $HAVE_VIPLAN && $HAVE_VLM; then
  DARGS=("${VLM_ARGS[@]}" "${VIPLAN_ARGS[@]}")
  run_step "ViPlan blocksworld render + VLM grounding" "VIPLAN-BLOCKSWORLD VLM GROUNDING OK" \
    "python -u examples/viplan_blocksworld_vlm_example.py"
elif ! $HAVE_VIPLAN; then
  SKIP+=("ViPlan blocksworld render/VLM — set VIPLAN_ROOT")
else
  SKIP+=("ViPlan blocksworld VLM grounding — set GAC")
fi

# ---- summary ---------------------------------------------------------------
echo; echo "############################################################"
echo "# SUMMARY"; echo "############################################################"
printf 'PASS (%d):\n' "${#PASS[@]}"; for x in "${PASS[@]}"; do echo "  PASS  $x"; done
printf 'SKIP (%d):\n' "${#SKIP[@]}"; for x in "${SKIP[@]}"; do echo "  SKIP  $x"; done
printf 'FAIL (%d):\n' "${#FAIL[@]}"; for x in "${FAIL[@]}"; do echo "  FAIL  $x"; done
if [ "${#FAIL[@]}" -eq 0 ]; then
  echo "ALL ATTEMPTED SIMULATORS PASSED"
else
  echo "SOME STEPS FAILED (RAM-heavy live steps need several GB free)"; exit 1
fi
