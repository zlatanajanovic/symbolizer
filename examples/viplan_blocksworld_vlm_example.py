"""End-to-end ViPlan Blocksworld VLM grounding example.

Blender-renders a ViPlan Blocksworld state, then VLM-grounds the rendered image
(objects + predicates) via the same provider path used everywhere else.

Prerequisites (see README "Table 6: ViPlan Benchmark"):
  - git clone https://github.com/merlerm/ViPlan.git $VIPLAN_ROOT && (cd $VIPLAN_ROOT && bash setup_blocksworld.sh)
  - build symbolizer-viplan:latest (Dockerfile.viplan), run with VIPLAN_ROOT mounted
  - a configured VLM provider (.env / env vars)

Run (inside the symbolizer-viplan container):
  VIPLAN_ROOT=/viplan PYTHONPATH=/viplan python examples/viplan_blocksworld_vlm_example.py

Verified live (Gemini Flash-Lite, CPU Blender): render ~1.7s -> 59KB PNG; grounded
7 objects (c1..c4 + purple/yellow/red blocks) + 18 atoms. Needs several GB RAM."""
import json, os, sys, tempfile
sys.path.insert(0, os.environ.get("VIPLAN_ROOT", "/viplan"))

from unified_planning.io import PDDLReader
from viplan.planning.blocksworld_simulator import BlocksworldSimulator
from symbolizer.ssr_parsing.run_eval_vlm import reinitialize_client
from symbolizer.ssr_parsing.ssr_parsing import retrieve_objects, retrieve_predicates

reinitialize_client()
VP = os.environ.get("VIPLAN_ROOT", "/viplan")
dom = f"{VP}/data/planning/blocksworld/domain.pddl"
_pdir = f"{VP}/data/planning/blocksworld/problems/simple"
prob = os.path.join(_pdir, sorted(f for f in os.listdir(_pdir) if f.endswith(".pddl"))[0])
print(f"domain={dom}\nproblem={prob}", flush=True)

# 1) Render the initial ViPlan state with Blender (CPU).
problem = PDDLReader().parse_problem(dom, prob)
env = BlocksworldSimulator(problem=problem, root_path=VP, fail_probability=0.0,
                           use_gpu_rendering=os.getenv("VIPLAN_USE_GPU_RENDERING", "0") == "1")
print("rendering state via Blender...", flush=True)
img = env.render()
img_dir = tempfile.mkdtemp()
img_path = os.path.join(img_dir, "viplan_state.png")
img.save(img_path)
print(f"RENDER OK -> {img_path} ({os.path.getsize(img_path)} bytes)", flush=True)

# 2) VLM-ground the rendered image (minimal dataset, zero-shot).
ds = {"problems": [{"problem_name": "viplan_bw", "domain_name": "blocksworld",
                    "domain_file": dom, "problem_file": prob,
                    "states": [{"id": 0, "image_path": img_path}]}]}
ds_path = os.path.join(img_dir, "ds.json")
json.dump(ds, open(ds_path, "w"))

print("VLM grounding objects on the rendered ViPlan image...", flush=True)
objs, _ = retrieve_objects(image_path=img_path, domain_name="blocksworld", dataset=ds,
                           num_examples=0, dataset_path=ds_path, number_answers=1, problem_id=0)
objects = objs[0].dict().get("objects", [])
print("GROUNDED OBJECTS:", [o.get("name") for o in objects], flush=True)

print("VLM grounding predicates...", flush=True)
preds, _ = retrieve_predicates(image_path=img_path, domain_name="blocksworld", objects=objs[0],
                               dataset=ds, num_examples=0, dataset_path=ds_path,
                               number_answers=1, problem_id=0)
atoms = preds[0].dict().get("grounded_predicates", [])
print("GROUNDED ATOMS:", len(atoms), flush=True)
print("VIPLAN-BLOCKSWORLD VLM GROUNDING OK", flush=True)
