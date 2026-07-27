"""Symbolizer-compatible ViPlan Blocksworld simulator.

This adapter uses ViPlan's native BlocksworldSimulator for transitions/rendering
and exposes the same interface expected by symbolizer search algorithms.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re

from PIL import Image
from unified_planning.io import PDDLReader
from unified_planning.model import Object
from unified_planning.plans.plan import ActionInstance

from symbolizer.ssr_parsing.run_eval_vlm import reinitialize_client
from symbolizer.ssr_parsing.ssr_parsing import retrieve_objects, retrieve_predicates


_REPO_ROOT = Path(__file__).resolve().parents[3]
# The ViPlan benchmark is an external checkout; point VIPLAN_ROOT at it to
# enable live ViPlan runs. The bundled symbolic ViPlan problems do not need it.
_VIPLAN_ROOT = Path(os.getenv("VIPLAN_ROOT", str(_REPO_ROOT / "experiments" / "benchmarks" / "ViPlan")))
_DATASETS_DIR = _REPO_ROOT / "experiments" / "data" / "datasets"

import sys
if str(_VIPLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIPLAN_ROOT))

try:
    from viplan.planning.blocksworld_simulator import BlocksworldSimulator  # noqa: E402
    from viplan.code_helpers import get_logger  # noqa: E402
    VIPLAN_AVAILABLE = True
except ImportError:
    BlocksworldSimulator = None
    get_logger = None
    VIPLAN_AVAILABLE = False

log = logging.getLogger(__name__)

_BLOCK_SYMBOL_TO_COLOR = {
    "G": "green",
    "Y": "yellow",
    "P": "purple",
    "B": "blue",
    "O": "orange",
    "R": "red",
}
_BLOCK_COLOR_TO_SYMBOL = {v: k.lower() for k, v in _BLOCK_SYMBOL_TO_COLOR.items()}


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text).strip("_")


@dataclass(frozen=True)
class ViPlanBWState:
    """Immutable state container for search algorithms."""

    snapshot: Tuple[Tuple[str, Tuple[Tuple[str, bool], ...]], ...]
    atoms: frozenset[str]
    goal: frozenset[str]


class ViPlanBlocksworldSimulator:
    """Search simulator backed by ViPlan's native Blocksworld environment."""

    @staticmethod
    def _enforce_phase2_gemini_logprobs_safety() -> None:
        """Prevent Phase-2 Gemini runs from inheriting unsupported logprobs mode."""
        if os.getenv("MODEL_TO_USE", "").strip().lower() != "gemini":
            return
        if os.getenv("PHASE2_ALLOW_GEMINI_LOGPROBS", "false").strip().lower() == "true":
            return
        if os.getenv("USE_LOGPROBS", "false").strip().lower() == "true":
            os.environ["USE_LOGPROBS"] = "false"
            os.environ["TOP_LOGPROBS"] = "0"

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        state_source: str = "symbolic",
        dataset_path: Optional[str] = None,
        num_examples: int = 3,
    ) -> None:
        if not VIPLAN_AVAILABLE:
            raise ImportError(
                "ViPlan benchmark not found. Clone it and set VIPLAN_ROOT to the "
                "checkout (needed only for live ViPlan runs; the bundled symbolic "
                "ViPlan problems do not need it)."
            )
        self.domain_file = str(domain_file)
        self.problem_file = str(problem_file)
        self.state_source = state_source.strip().lower()
        if self.state_source not in {"symbolic", "vlm"}:
            raise ValueError("state_source must be one of: 'symbolic', 'vlm'")

        self.dataset_path = dataset_path
        self.num_examples = int(num_examples)
        self._atom_probabilities: Dict[str, float] = {}
        self._last_usage: Dict[str, Any] = {
            "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
        }
        self._usage_totals: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_content_tokens": 0,
            "calls_objects": 0,
            "calls_predicates": 0,
            "objects_reused": 0,
            "cache_hits": 0,
            "shared_cache_hits": 0,
            "retry_object_attempts": 0,
            "retry_predicate_attempts": 0,
            "retry_object_successes": 0,
            "retry_predicate_successes": 0,
        }
        self._state_cache: Dict[Tuple[Tuple[str, Tuple[Tuple[str, bool], ...]], ...], Dict[str, Any]] = {}
        self._render_cache_dir = Path(tempfile.mkdtemp(prefix="viplan_bw_obs_"))
        self._obs_cache: Dict[str, Tuple[Set[str], Dict[str, float]]] = {}
        self._state_observation_count = 0
        self._object_refresh_first_n = max(0, int(os.getenv("VLM_OBJECT_REFRESH_FIRST_N", "5") or 5))
        self._object_refresh_period = max(0, int(os.getenv("VLM_OBJECT_REFRESH_PERIOD", "50") or 50))
        self._obs_retry_attempts = max(1, int(os.getenv("VLM_OBS_RETRY_ATTEMPTS", "3") or 3))
        self._obs_retry_backoff_s = max(0.0, float(os.getenv("VLM_OBS_RETRY_BACKOFF_S", "0.25") or 0.25))
        self._accumulate_objects = os.getenv("VLM_ACCUMULATE_OBJECTS", "false").strip().lower() == "true"
        self._object_bootstrap_states = max(1, int(os.getenv("VLM_OBJECT_BOOTSTRAP_STATES", "5") or 5))
        self._object_bootstrap_threshold = float(os.getenv("VLM_OBJECT_BOOTSTRAP_THRESHOLD", "0.75") or 0.75)
        if self._object_bootstrap_threshold < 0.0:
            self._object_bootstrap_threshold = 0.0
        if self._object_bootstrap_threshold > 1.0:
            self._object_bootstrap_threshold = 1.0
        self._last_objects_model = None
        self._seen_objects_by_name: Dict[str, Any] = {}
        self._last_pred_object_names: Set[str] = set()
        self._object_bootstrap_counts: Dict[str, int] = {}
        self._object_bootstrap_type_by_name: Dict[str, str] = {}
        self._object_bootstrap_obs = 0
        self._model_key = (
            os.getenv("MODEL_NAME", "").strip()
            or os.getenv("MODEL_TO_USE", "").strip()
            or "unknown_model"
        )
        storage_root = Path(
            os.getenv(
                "VLM_PLANNING_STORAGE_DIR",
                str(
                    Path(__file__).resolve().parents[4]
                    / "experiments"
                    / "final_clean"
                    / "phase2_planning"
                    / "storage"
                    / "vlm_image_cache"
                ),
            )
        )
        self._shared_storage_dir = (
            storage_root
            / "viplan_blocksworld"
            / _safe_slug(self._model_key)
            / f"nex_{self.num_examples}"
        )
        self._shared_storage_dir.mkdir(parents=True, exist_ok=True)

        if self.state_source == "vlm":
            if not self.dataset_path:
                raise ValueError("dataset_path is required when state_source='vlm'")
            self._enforce_phase2_gemini_logprobs_safety()
            with open(self.dataset_path, "r", encoding="utf-8") as fh:
                self._dataset = json.load(fh)
            self._dataset = self._alias_dataset_examples(self._dataset)
            # Map current problem to its dataset index so few-shot selection and
            # problem-local context match the active task.
            self._problem_id = 0
            # In phase2 data layout, filename is always "problem.pddl"; use
            # directory name as the actual sample/problem identifier.
            p_stem = Path(self.problem_file).parent.name
            ds_probs = list(self._dataset.get("problems", []))
            pref = f"viplan_blocksworld_{p_stem}"
            for i, pr in enumerate(ds_probs):
                pn = str(pr.get("problem_name", ""))
                if pn == p_stem or pn == pref or pn.endswith(f"_{p_stem}"):
                    self._problem_id = i
                    break
            reinitialize_client()
        else:
            self._dataset = None
            self._problem_id = 0

        reader = PDDLReader()
        self._problem = reader.parse_problem(self.domain_file, self.problem_file)
        # Default to CPU rendering for portability (no CUDA needed). Set
        # VIPLAN_USE_GPU_RENDERING=1 on a CUDA host to accelerate Blender renders.
        _use_gpu = os.getenv("VIPLAN_USE_GPU_RENDERING", "0").strip() == "1"
        self._env = BlocksworldSimulator(
            problem=self._problem,
            root_path=str(_VIPLAN_ROOT),
            logger=get_logger("warning"),
            fail_probability=0.0,
            use_gpu_rendering=_use_gpu,
        )

        self._goal_atoms = frozenset(self._goal_atoms_from_problem())
        self._all_actions = self._enumerate_all_actions()

    def __del__(self) -> None:
        try:
            if hasattr(self, "_render_cache_dir") and self._render_cache_dir.exists():
                for p in self._render_cache_dir.glob("*"):
                    p.unlink(missing_ok=True)
                self._render_cache_dir.rmdir()
        except Exception:
            pass

    @staticmethod
    def _lit_str(name: str, args: Iterable[str]) -> str:
        args_tuple = tuple(str(a) for a in args)
        return f"{name}({','.join(args_tuple)})" if args_tuple else f"{name}()"

    def _goal_atoms_from_problem(self) -> List[str]:
        atoms: List[str] = []
        goals = self._problem.goals[0] if self._problem.goals else None
        if goals is None:
            return atoms
        if goals.is_and():
            goal_nodes = goals.args
        else:
            goal_nodes = [goals]
        for g in goal_nodes:
            if g.is_not():
                # Keep positive-goal semantics used in existing pipeline.
                continue
            if not g.is_fluent_exp():
                continue
            name = g.fluent().name
            args = [str(a) for a in g.args]
            atoms.append(self._lit_str(name, args))
        return atoms

    def _state_to_snapshot(self, state_dict: Dict[str, Any]) -> Tuple[Tuple[str, Tuple[Tuple[str, bool], ...]], ...]:
        rows: List[Tuple[str, Tuple[Tuple[str, bool], ...]]] = []
        for fluent in sorted(state_dict):
            inner = tuple(sorted((str(k), bool(v)) for k, v in state_dict[fluent].items()))
            rows.append((fluent, inner))
        return tuple(rows)

    def _snapshot_to_state_dict(self, snapshot: Tuple[Tuple[str, Tuple[Tuple[str, bool], ...]], ...]) -> Dict[str, Any]:
        out = copy.deepcopy(self._env.state)
        for fluent, entries in snapshot:
            out[fluent].clear()
            for key, value in entries:
                out[fluent][key] = bool(value)
        return out

    def _atoms_from_state_dict(self, state_dict: Dict[str, Any]) -> Set[str]:
        atoms: Set[str] = set()
        for fluent, entries in state_dict.items():
            for key, value in entries.items():
                if not value:
                    continue
                atoms.add(self._lit_str(fluent, key.split(",") if key else []))
        return atoms

    def _build_state(self, state_dict: Dict[str, Any]) -> ViPlanBWState:
        snap = self._state_to_snapshot(state_dict)
        self._state_cache[snap] = copy.deepcopy(state_dict)
        atoms = frozenset(self._atoms_from_state_dict(state_dict))
        return ViPlanBWState(snapshot=snap, atoms=atoms, goal=self._goal_atoms)

    def _enumerate_all_actions(self) -> List[ActionInstance]:
        actions: List[ActionInstance] = []
        for action in self._problem.actions:
            obj_lists: List[List[Object]] = []
            for p in action.parameters:
                tname = str(p.type)
                obj_lists.append(list(self._env.all_objects[tname]))
            if not obj_lists:
                actions.append(ActionInstance(action, []))
                continue
            # Cartesian product without itertools to keep dependency surface tiny.
            indices = [0] * len(obj_lists)
            while True:
                params = [obj_lists[i][indices[i]] for i in range(len(obj_lists))]
                actions.append(ActionInstance(action, params))
                j = len(indices) - 1
                while j >= 0:
                    indices[j] += 1
                    if indices[j] < len(obj_lists[j]):
                        break
                    indices[j] = 0
                    j -= 1
                if j < 0:
                    break
        return actions

    def reset(self) -> ViPlanBWState:
        self._env.reset()
        self._atom_probabilities = {}
        self._last_usage = {
            "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
        }
        self._usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_content_tokens": 0,
            "calls_objects": 0,
            "calls_predicates": 0,
            "objects_reused": 0,
            "cache_hits": 0,
            "shared_cache_hits": 0,
            "retry_object_attempts": 0,
            "retry_predicate_attempts": 0,
            "retry_object_successes": 0,
            "retry_predicate_successes": 0,
        }
        self._obs_cache.clear()
        self._state_observation_count = 0
        self._last_objects_model = None
        self._seen_objects_by_name.clear()
        self._last_pred_object_names = set()
        self._object_bootstrap_counts.clear()
        self._object_bootstrap_type_by_name.clear()
        self._object_bootstrap_obs = 0
        return self._build_state(self._env.state)

    @staticmethod
    def _object_names_from_model(objects_model: Any) -> Set[str]:
        names: Set[str] = set()
        for obj in getattr(objects_model, "objects", []):
            name = getattr(obj, "name", None)
            if name is None:
                continue
            names.add(str(name).lower())
        return names

    @staticmethod
    def _normalize_block_name_for_vlm(name: str) -> str:
        n = str(name).strip()
        if len(n) == 1 and n.upper() in _BLOCK_SYMBOL_TO_COLOR:
            return _BLOCK_SYMBOL_TO_COLOR[n.upper()]
        return n.lower()

    @staticmethod
    def _normalize_pred_arg_to_symbol(name: str) -> str:
        n = str(name).strip().lower()
        if n.endswith("_block"):
            n = n[:-6]
        if n in _BLOCK_COLOR_TO_SYMBOL:
            return _BLOCK_COLOR_TO_SYMBOL[n]
        m = re.fullmatch(r"column[_-]?(\d+)", n)
        if m:
            return f"c{m.group(1)}"
        return n

    @staticmethod
    def _alias_name_for_examples(name: Any, type_hint: Optional[str] = None) -> Any:
        s = str(name).strip()
        th = (type_hint or "").strip().lower()
        if th == "block":
            if len(s) == 1 and s.upper() in _BLOCK_SYMBOL_TO_COLOR:
                return _BLOCK_SYMBOL_TO_COLOR[s.upper()]
            return s.lower()
        if th == "column":
            m = re.fullmatch(r"[cC](\d+)", s)
            return f"c{m.group(1)}" if m else s.lower()

        m = re.fullmatch(r"[cC](\d+)", s)
        if m:
            return f"c{m.group(1)}"
        if len(s) == 1 and s.upper() in _BLOCK_SYMBOL_TO_COLOR:
            return _BLOCK_SYMBOL_TO_COLOR[s.upper()]
        return s.lower()

    @classmethod
    def _alias_examples_payload(cls, payload: Any) -> Any:
        if isinstance(payload, dict):
            out: Dict[str, Any] = {}
            type_hint = payload.get("type") if isinstance(payload.get("type"), str) else None
            for k, v in payload.items():
                if k == "name" and isinstance(v, str):
                    out[k] = cls._alias_name_for_examples(v, type_hint=type_hint)
                else:
                    out[k] = cls._alias_examples_payload(v)
            return out
        if isinstance(payload, list):
            return [cls._alias_examples_payload(x) for x in payload]
        return payload

    @classmethod
    def _alias_dataset_examples(cls, dataset: Dict[str, Any]) -> Dict[str, Any]:
        patched = copy.deepcopy(dataset)
        for pr in patched.get("problems", []):
            for st in pr.get("states", []):
                for field in ("all_objects", "atoms", "goal_predicates"):
                    raw = st.get(field)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    try:
                        parsed = json.loads(raw)
                        st[field] = json.dumps(cls._alias_examples_payload(parsed))
                    except Exception:
                        continue
        return patched

    def _merge_objects_into_memory(self, objects_model: Optional[Any]) -> Optional[Any]:
        objs = list(getattr(objects_model, "objects", []) or [])
        for obj in objs:
            name = str(getattr(obj, "name", "")).strip()
            if not name:
                continue
            self._seen_objects_by_name[name.lower()] = obj
        if not self._seen_objects_by_name:
            return self._last_objects_model
        merged = [self._seen_objects_by_name[k] for k in sorted(self._seen_objects_by_name)]
        base = self._last_objects_model or objects_model
        if base is None:
            return None
        if hasattr(base, "model_copy"):
            return base.model_copy(update={"objects": merged})
        cls = type(base)
        payload = {"objects": [{"name": getattr(o, "name", ""), "type": getattr(o, "type", "")} for o in merged]}
        if hasattr(cls, "model_validate"):
            return cls.model_validate(payload)
        if hasattr(cls, "parse_obj"):
            return cls.parse_obj(payload)
        return base

    def _bootstrap_object_name(self, obj: Any) -> str:
        raw_name = str(getattr(obj, "name", "")).strip()
        raw_type = str(getattr(obj, "type", "")).strip().lower()
        if raw_type == "block":
            return self._normalize_block_name_for_vlm(raw_name)
        m = re.fullmatch(r"column[_-]?(\d+)", raw_name.strip().lower())
        if m:
            return f"c{m.group(1)}"
        return raw_name.lower()

    def _bootstrap_object_type(self, obj: Any) -> str:
        return str(getattr(obj, "type", "")).strip().lower()

    def _build_objects_model_from_bootstrap(self, base_model: Any) -> Any:
        if self._object_bootstrap_obs <= 0:
            return base_model
        min_count = max(1, math.ceil(self._object_bootstrap_obs * self._object_bootstrap_threshold))
        selected: List[Any] = []
        for name in sorted(self._object_bootstrap_counts):
            if self._object_bootstrap_counts[name] >= min_count:
                selected.append({
                    "name": name,
                    "type": self._object_bootstrap_type_by_name.get(name, "object"),
                })
        if not selected:
            return base_model
        cls = type(base_model)
        payload = {"objects": selected}
        if hasattr(cls, "model_validate"):
            return cls.model_validate(payload)
        if hasattr(cls, "parse_obj"):
            return cls.parse_obj(payload)
        if hasattr(base_model, "model_copy"):
            return base_model.model_copy(update=payload)
        return base_model

    def _update_bootstrap_objects(self, objects_model: Optional[Any]) -> Optional[Any]:
        if objects_model is None:
            return None
        if self._object_bootstrap_obs >= self._object_bootstrap_states:
            return self._build_objects_model_from_bootstrap(objects_model)
        current_names: Set[str] = set()
        for obj in getattr(objects_model, "objects", []) or []:
            name = self._bootstrap_object_name(obj)
            if not name:
                continue
            current_names.add(name)
            self._object_bootstrap_type_by_name[name] = self._bootstrap_object_type(obj)
        for name in current_names:
            self._object_bootstrap_counts[name] = self._object_bootstrap_counts.get(name, 0) + 1
        self._object_bootstrap_obs += 1
        return self._build_objects_model_from_bootstrap(objects_model)

    def get_actions(self, state: ViPlanBWState) -> List[ActionInstance]:
        state_dict = self._state_cache.get(state.snapshot) or self._snapshot_to_state_dict(state.snapshot)
        legal: List[ActionInstance] = []
        for a in self._all_actions:
            if self._env.is_action_legal(a, state=state_dict):
                legal.append(a)
        return sorted(legal, key=str, reverse=True)

    def step(self, state: ViPlanBWState, action: ActionInstance) -> ViPlanBWState:
        state_dict = self._state_cache.get(state.snapshot) or self._snapshot_to_state_dict(state.snapshot)
        self._env.set_state(copy.deepcopy(state_dict))
        ok, _ = self._env.apply_action(action, can_fail=False)
        if not ok:
            return state
        return self._build_state(self._env.state)

    def is_goal(self, state: ViPlanBWState) -> bool:
        return state.goal.issubset(state.atoms)

    def _render_state_to_image(self, state: ViPlanBWState) -> str:
        state_dict = self._state_cache.get(state.snapshot) or self._snapshot_to_state_dict(state.snapshot)
        self._env.set_state(copy.deepcopy(state_dict))
        img = self._env.render(label_columns=True)
        if img is None:
            raise RuntimeError("ViPlan render returned no image")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        image_hash = hashlib.md5("|".join(sorted(state.atoms)).encode("utf-8")).hexdigest()[:16]
        out = self._render_cache_dir / f"obs_{image_hash}.png"
        img.save(out)
        return str(out)

    def _image_md5(self, image_path: str) -> str:
        return hashlib.md5(Path(image_path).read_bytes()).hexdigest()

    def _shared_entry_path(self, image_md5: str) -> Path:
        return self._shared_storage_dir / f"{image_md5}.json"

    def _write_shared_entry(
        self,
        *,
        image_md5: str,
        atoms: Set[str],
        probs: Dict[str, float],
        object_names: Set[str],
    ) -> None:
        path = self._shared_entry_path(image_md5)
        if path.exists():
            return
        payload = {
            "atoms": sorted(str(a) for a in atoms),
            "atom_probs": {str(k): float(v) for k, v in probs.items()},
            "object_names": sorted(str(x) for x in object_names),
            "domain": "viplan_blocksworld",
            "model": self._model_key,
            "num_examples": int(self.num_examples),
            "image_md5": image_md5,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _pred_entry_to_atom(pred_entry: Any) -> Optional[str]:
        pname = getattr(pred_entry, "predicate_type", getattr(pred_entry, "predicate", ""))
        if not pname:
            return None
        args: List[str] = []
        for key in ("x", "y", "z"):
            val = getattr(pred_entry, key, None)
            if val is None:
                continue
            arg_name = getattr(val, "name", str(val))
            if str(arg_name).lower() == "empty":
                continue
            # Map color aliases (orange/yellow/...) and column variants back to symbolic names.
            args.append(ViPlanBlocksworldSimulator._normalize_pred_arg_to_symbol(str(arg_name)))
        return f"{str(pname).lower()}({','.join(args)})" if args else f"{str(pname).lower()}()"

    def make_observation(self, state: ViPlanBWState) -> Dict[str, Set[str]]:
        if self.state_source == "symbolic":
            self._atom_probabilities = {a: 1.0 for a in state.atoms}
            self._last_pred_object_names = set()
            return {"atoms": set(state.atoms)}

        state_hash = hashlib.md5("|".join(sorted(state.atoms)).encode("utf-8")).hexdigest()[:16]
        cached = self._obs_cache.get(state_hash)
        if cached is not None:
            atoms_cached, probs_cached = cached
            self._atom_probabilities = dict(probs_cached)
            self._last_usage = {
                "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
                "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
                "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            }
            self._usage_totals["cache_hits"] += 1
            self._last_pred_object_names = (
                self._object_names_from_model(self._last_objects_model)
                if self._last_objects_model is not None else set()
            )
            return {"atoms": set(atoms_cached)}

        image_path = self._render_state_to_image(state)
        image_md5 = self._image_md5(image_path)
        self._state_observation_count += 1

        shared_path = self._shared_entry_path(image_md5)
        if shared_path.exists():
            shared_entry = json.loads(shared_path.read_text(encoding="utf-8"))
            atoms_shared = set(str(a) for a in shared_entry.get("atoms", []))
            probs_shared_raw = shared_entry.get("atom_probs", {})
            probs_shared = {str(k): float(v) for k, v in probs_shared_raw.items()}
            self._obs_cache[state_hash] = (set(atoms_shared), dict(probs_shared))
            self._atom_probabilities = dict(probs_shared)
            self._last_pred_object_names = set(str(x).lower() for x in shared_entry.get("object_names", []))
            self._last_usage = {
                "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
                "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
                "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_content_tokens": 0},
            }
            self._usage_totals["cache_hits"] += 1
            self._usage_totals["shared_cache_hits"] += 1
            return {"atoms": set(atoms_shared)}

        should_refresh = (
            self._last_objects_model is None
            or self._state_observation_count <= self._object_refresh_first_n
            or (
                self._object_refresh_period > 0
                and (self._state_observation_count - self._object_refresh_first_n) % self._object_refresh_period == 0
            )
        )

        objects_model = self._last_objects_model
        obj_usage: Dict[str, Any] = {}
        if should_refresh:
            objects_list: List[Any] = []
            for attempt_idx in range(self._obs_retry_attempts):
                obj_usage_attempt: Dict[str, Any] = {}
                objects_list, _ = retrieve_objects(
                    image_path=image_path,
                    domain_name="blocksworld",
                    dataset=self._dataset,
                    dataset_path=self.dataset_path,
                    num_examples=self.num_examples,
                    problem_id=self._problem_id,
                    _out_usage=obj_usage_attempt,
                )
                if obj_usage_attempt:
                    obj_usage = obj_usage_attempt
                objects_model = objects_list[0] if objects_list else None
                if objects_model is not None:
                    if attempt_idx > 0:
                        self._usage_totals["retry_object_successes"] += 1
                    break
                if attempt_idx + 1 < self._obs_retry_attempts:
                    self._usage_totals["retry_object_attempts"] += 1
                    if self._obs_retry_backoff_s > 0.0:
                        time.sleep(self._obs_retry_backoff_s)
            if objects_model is not None:
                objects_model = self._update_bootstrap_objects(objects_model)
                if self._accumulate_objects:
                    self._last_objects_model = self._merge_objects_into_memory(objects_model) or objects_model
                else:
                    self._last_objects_model = objects_model
            self._usage_totals["calls_objects"] += 1
        else:
            self._usage_totals["objects_reused"] += 1

        if objects_model is None:
            self._atom_probabilities = {}
            self._obs_cache[state_hash] = (set(), {})
            self._last_pred_object_names = set()
            return {"atoms": set()}
        objects_model = self._last_objects_model or objects_model
        self._last_pred_object_names = self._object_names_from_model(objects_model)
        pred_usage: Dict[str, Any] = {}
        preds_list: List[Any] = []
        pred_probs_list: List[Any] = []
        for attempt_idx in range(self._obs_retry_attempts):
            pred_usage_attempt: Dict[str, Any] = {}
            preds_list, pred_probs_list = retrieve_predicates(
                image_path=image_path,
                domain_name="blocksworld",
                objects=objects_model,
                dataset=self._dataset,
                dataset_path=self.dataset_path,
                num_examples=self.num_examples,
                problem_id=self._problem_id,
                _out_usage=pred_usage_attempt,
            )
            if pred_usage_attempt:
                pred_usage = pred_usage_attempt
            preds_model = preds_list[0] if preds_list else None
            if preds_model is not None:
                if attempt_idx > 0:
                    self._usage_totals["retry_predicate_successes"] += 1
                break
            if attempt_idx + 1 < self._obs_retry_attempts:
                self._usage_totals["retry_predicate_attempts"] += 1
                if self._obs_retry_backoff_s > 0.0:
                    time.sleep(self._obs_retry_backoff_s)
        self._usage_totals["calls_predicates"] += 1
        preds_model = preds_list[0] if preds_list else None
        pred_probs = pred_probs_list[0] if pred_probs_list else []

        atoms: Set[str] = set()
        probs: Dict[str, float] = {}
        raw_preds = list(getattr(preds_model, "grounded_predicates", []) or [])
        dropped_predicates = 0
        for gp in raw_preds:
            atom = self._pred_entry_to_atom(gp)
            if atom is None:
                dropped_predicates += 1
                continue
            atoms.add(atom)
            probs.setdefault(atom, 1.0)
        for entry in pred_probs:
            pred = entry.get("predicate")
            if pred is None:
                continue
            atom = self._pred_entry_to_atom(pred)
            if atom is None:
                continue
            probs[atom] = float(entry.get("prob", 1.0))

        if dropped_predicates > 0:
            log.warning(
                "ViPlan BW predicate mapping dropped %d/%d predictions for state hash=%s",
                dropped_predicates,
                len(raw_preds),
                state_hash[:12],
            )
        if raw_preds and not atoms:
            log.warning(
                "All predicted predicates were filtered out for ViPlan BW state hash=%s (raw=%d).",
                state_hash[:12],
                len(raw_preds),
            )

        obj_prompt = int(obj_usage.get("prompt_tokens", 0) or 0)
        obj_completion = int(obj_usage.get("completion_tokens", 0) or 0)
        obj_total = int(obj_usage.get("total_tokens", 0) or 0)
        obj_cached = int(obj_usage.get("cached_content_tokens", 0) or 0)
        pred_prompt = int(pred_usage.get("prompt_tokens", 0) or 0)
        pred_completion = int(pred_usage.get("completion_tokens", 0) or 0)
        pred_total = int(pred_usage.get("total_tokens", 0) or 0)
        pred_cached = int(pred_usage.get("cached_content_tokens", 0) or 0)
        total_prompt = obj_prompt + pred_prompt
        total_completion = obj_completion + pred_completion
        total_tokens = obj_total + pred_total
        total_cached = obj_cached + pred_cached
        any_usage = bool(obj_usage.get("available", False) or pred_usage.get("available", False))
        self._last_usage = {
            "objects": {
                "available": bool(obj_usage.get("available", False)),
                "prompt_tokens": obj_prompt,
                "completion_tokens": obj_completion,
                "total_tokens": obj_total,
                "cached_content_tokens": obj_cached,
            },
            "predicates": {
                "available": bool(pred_usage.get("available", False)),
                "prompt_tokens": pred_prompt,
                "completion_tokens": pred_completion,
                "total_tokens": pred_total,
                "cached_content_tokens": pred_cached,
            },
            "total": {
                "available": any_usage,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "cached_content_tokens": total_cached,
            },
        }
        self._usage_totals["prompt_tokens"] += total_prompt
        self._usage_totals["completion_tokens"] += total_completion
        self._usage_totals["total_tokens"] += total_tokens
        self._usage_totals["cached_content_tokens"] += total_cached

        # Safety guard: empty observations for non-empty symbolic states break A* guidance.
        if not atoms and state.atoms:
            raise RuntimeError(
                "ViPlan VLM observation returned empty predicates for a non-empty state. "
                "This usually indicates incompatible Gemini logprobs setup. "
                "Ensure USE_LOGPROBS=false and TOP_LOGPROBS=0 for Phase-2."
            )

        self._atom_probabilities = probs
        self._obs_cache[state_hash] = (set(atoms), dict(probs))
        self._write_shared_entry(
            image_md5=image_md5,
            atoms=atoms,
            probs=probs,
            object_names=self._last_pred_object_names,
        )
        return {"atoms": atoms}

    def get_state_probability(self, state: ViPlanBWState) -> Dict[str, float]:
        if not self._atom_probabilities:
            return {a: 1.0 for a in state.atoms}
        return self._atom_probabilities

    def get_last_usage(self) -> Dict[str, Any]:
        return self._last_usage

    def get_usage_totals(self) -> Dict[str, int]:
        return dict(self._usage_totals)

    def get_last_observation_diagnostics(self) -> Dict[str, Any]:
        return {
            "pred_object_names": sorted(self._last_pred_object_names),
        }

    def state_to_atoms(self, state: ViPlanBWState) -> Dict[str, Set[str]]:
        return {"atoms": set(state.atoms)}

    def atoms_to_state(self, atoms: Dict[str, Set[str]]) -> frozenset[str]:
        return frozenset(atoms.get("atoms", set()))
