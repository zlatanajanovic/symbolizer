# Reproducing the Paper Tables

This is a **code-only** release: the experiment code and all input data needed to
reproduce every table are included, but our precomputed result files are not. You
reproduce the numbers by running the experiments and comparing against the values
reported in the paper.

## Offline planning tables (4–5, symbolic ViPlan rows of 6) — no API, no cost

```bash
# Re-execute the shipped symbolic planner on the bundled problems and report
# solved / total per domain (A* and A*+Novelty = the paper's planning method):
python -m experiments.regenerate_all_tables --smoke

# Or run the planner directly with full per-domain output:
python -m experiments.run_planning_benchmark --data_dir experiments/data/planning
```

The reported planning method is **A\*** with a **goal-count heuristic** and
**width-based novelty as a tie-breaker** (the `A*+Nov` column of
`run_planning_benchmark.py`; plain `A*` is the no-tiebreak variant). The `IWK`,
`SIW`, and `BFWS` columns in the same harness are ablations and were not used for
any reported number. `--smoke` confirms `A*` and `A*+Nov` solve 14/14 of the
offline-runnable problems (ViPlan-Blocks is skipped unless `VIPLAN_ROOT` is set —
its transition model lives in the external ViPlan benchmark).

## Grounding tables (1–3) and VLM-in-the-loop rows — live, paid

These make live VLM calls. Configure `.env` (provider + credentials) first, then:

```bash
python -m experiments.regenerate_all_tables --print-live-commands   # prints every recipe
```

| Table | What | How to reproduce |
|-------|------|------------------|
| 1 | Object grounding F1 | `run_grounding_eval --dataset {pddlgym,real,vilain}` |
| 2 | Predicate grounding F1 | same as Table 1 |
| 3 | Goal grounding F1 | same as Table 1 (the real-image goal cell needs instruction files not released for the real set — see `../SIMULATORS.md`) |
| 4 | ProDG/PDDLGym/PyBullet planning | `run_planning_benchmark` (offline) + `run_search_comparison` (live VLM-grounded) |
| 5 | Own-domain planning | `run_planning_benchmark` (offline) + `run_search_comparison` (live VLM-grounded) |
| 6 | ViPlan | `run_planning_benchmark` with `VIPLAN_ROOT` (symbolic) + `examples/viplan_blocksworld_vlm_example.py` (render + grounding) |

A fresh live re-run is a new sample and may differ slightly from the paper's
printed values due to VLM nondeterminism; compare against the paper's tables.

## Docker

```bash
docker build -t symbolizer-release:latest .
docker run --rm -v "$PWD":/workspace -w /workspace symbolizer-release:latest \
    python -m experiments.regenerate_all_tables --smoke
```

This needs no API keys — the checked-in planning problems ship with the image, so
the offline planner reproduction runs out of the box.
