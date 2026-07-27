"""VLM-based PDDL simulator for file-based Phase-2 style planning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pddlgym import make as pddlgym_make
from pddlgym.structs import Literal, Predicate, State, Type, TypedEntity

from .pddl_file_simulator import PddlFileSimulator
from symbolizer.ssr_parsing.ssr_parsing import retrieve_objects, retrieve_predicates
from symbolizer.ssr_parsing.run_eval_vlm import client as _vlm_client, reinitialize_client

log = logging.getLogger(__name__)

_DOMAIN_CONFIG: Dict[str, tuple[str, str, str]] = {
    "blocks": ("blocks_ssr.json", "blocks", "PDDLEnvBlocks-v0"),
    # ViLaIn uses the directory/domain key "blocksworld" for the same blocks domain.
    "blocksworld": ("blocksworld_ssr.json", "blocksworld", "PDDLEnvBlocks-v0"),
    "hanoi": ("hanoi_ssr.json", "hanoi", "PDDLEnvHanoi-v0"),
    "hanoi_color": ("hanoi_color_ssr.json", "hanoi_color", "PDDLEnvHanoi_color-v0"),
    # ViLaIn cooking uses the kitchen renderer backend, but a distinct SSR dataset/domain key.
    "cooking": ("cooking_ssr.json", "cooking", ""),
    "kitchen_worlds": ("kitchen_worlds_ssr.json", "kitchen_worlds", ""),
    "household": ("household_ssr.json", "household", ""),
}

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATASETS_DIR = _REPO_ROOT / "experiments" / "final_clean" / "phase1_ssr" / "datasets"


def _concat_images_horizontally(images: list[Image.Image]) -> Image.Image:
    valid = [im.convert("RGB") for im in images if im is not None]
    if not valid:
        raise ValueError("No images to concatenate.")
    total_w = sum(im.width for im in valid)
    max_h = max(im.height for im in valid)
    canvas = Image.new("RGB", (total_w, max_h))
    x = 0
    for im in valid:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def _pybullet_env_setup() -> None:
    coast_block = _REPO_ROOT / "experiments" / "coast_exp" / "coast" / "envs" / "block"
    coast_root = _REPO_ROOT / "experiments" / "coast_exp"
    pb_planning = _REPO_ROOT / "experiments" / "coast_exp" / "pybullet-planning"
    for p in (coast_block, coast_root, pb_planning):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


class VLMPddlFileSimulator(PddlFileSimulator):
    """PddlFileSimulator with VLM-based observations via the SSR pipeline."""

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
        suite: str,
        domain_key: str,
        num_examples: int = 3,
    ) -> None:
        super().__init__(domain_file, problem_file)

        self._suite = suite.lower()
        self._num_examples = num_examples
        self._enforce_phase2_gemini_logprobs_safety()
        self._atom_probabilities: Dict[Literal, float] = {}
        self._last_logprobs: Optional[Dict] = None
        self._last_usage: Dict[str, Any] = {
            "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._usage_totals: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls_objects": 0,
            "calls_predicates": 0,
            "cache_hits": 0,
            "objects_reused": 0,
        }
        self._state_observation_count = 0
        self._object_refresh_first_n = int(os.getenv("VLM_OBJECT_REFRESH_FIRST_N", "5"))
        self._object_refresh_period = int(os.getenv("VLM_OBJECT_REFRESH_PERIOD", "50"))
        self._object_stability_threshold = float(os.getenv("VLM_OBJECT_STABILITY_THRESHOLD", "0.75"))
        # Opt-in accumulation; enable by setting VLM_ACCUMULATE_OBJECTS=true.
        self._accumulate_objects = os.getenv("VLM_ACCUMULATE_OBJECTS", "false").strip().lower() == "true"
        self._object_call_count = 0
        self._object_presence_counts: Dict[tuple[str, str], int] = {}
        self._object_canonical: Dict[tuple[str, str], Dict[str, str]] = {}
        self._last_objects_instance: Any = None
        self._stable_objects_instance: Any = None
        self._stable_object_keys: Set[tuple[str, str]] = set()

        cfg = _DOMAIN_CONFIG.get(domain_key.lower())
        if cfg is None:
            raise ValueError(f"VLMPddlFileSimulator: unknown domain key '{domain_key}'. Supported: {list(_DOMAIN_CONFIG)}")
        dataset_suffix, self._ssr_domain_name, pddlgym_env_name = cfg

        cache_root = os.getenv(
            "VLM_OBS_CACHE_DIR",
            str(_REPO_ROOT / "experiments" / "final_clean" / "phase2_planning" / "batch_cache" / "vlm_obs_cache"),
        )
        cache_sig_src = "|".join(
            [
                str(domain_file),
                str(problem_file),
                str(self._ssr_domain_name),
                str(self._num_examples),
                str(os.getenv("MODEL_TO_USE", "")),
                str(os.getenv("MODEL_NAME", "")),
            ]
        )
        self._cache_sig = hashlib.sha1(cache_sig_src.encode("utf-8")).hexdigest()[:16]
        self._cache_dir = Path(cache_root) / self._cache_sig
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._dataset_path = str(_DATASETS_DIR / f"{suite}_{dataset_suffix}")
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(f"SSR dataset not found: {self._dataset_path}")
        with open(self._dataset_path) as fh:
            self._dataset = json.load(fh)

        self._kitchen_renderer = None
        self._kitchen_module = None
        self._kitchen_scene_dir: Optional[str] = None
        self._pybullet_hanoi_renderer = False
        self._kitchen_show_robot_panel = (
            os.getenv("PHASE2_KITCHEN_SHOW_ROBOT_PANEL", "true").strip().lower() == "true"
        )

        self._problem_id = self._resolve_problem_id(problem_file)

        self._render_fn = None
        if self._ssr_domain_name in {"kitchen_worlds", "cooking"}:
            try:
                self._build_kitchen_renderer()
                log.info("VLMPddlFileSimulator: kitchen renderer ready")
            except Exception as exc:
                log.warning("VLMPddlFileSimulator: kitchen renderer unavailable: %s", exc)
        elif self._suite == "pybullet" and self._ssr_domain_name == "hanoi":
            try:
                _pybullet_env_setup()
                from test_hanoi import HanoiSimulator  # noqa: F401

                self._pybullet_hanoi_renderer = True
                log.info("VLMPddlFileSimulator: renderer ready (pybullet_hanoi_simulator)")
            except Exception as exc:
                log.warning("VLMPddlFileSimulator: pybullet hanoi renderer unavailable: %s", exc)
        else:
            try:
                renderer_env = pddlgym_make(pddlgym_env_name)
                renderer_env.fix_problem_index(0)
                renderer_env.reset()
                fn = renderer_env.unwrapped._render
                if callable(fn):
                    self._render_fn = fn
                    log.info("VLMPddlFileSimulator: renderer ready (%s)", pddlgym_env_name)
            except Exception as exc:
                log.warning("VLMPddlFileSimulator: renderer unavailable for %s: %s", pddlgym_env_name, exc)

        self._image_dir = tempfile.mkdtemp(prefix="vlm_phase2_obs_")

        if _vlm_client is None:
            reinitialize_client()

    def _resolve_problem_id(self, problem_file: str) -> int:
        problems = self._dataset.get("problems", [])
        if not problems:
            return 0

        problem_path = Path(problem_file).resolve()
        candidate_tokens = []
        for part in problem_path.parts:
            token = part.lower()
            if re.fullmatch(r"problem_\d+_sample_\d+", token):
                candidate_tokens.append(token)
        if problem_path.parent.name:
            candidate_tokens.append(problem_path.parent.name.lower())
        candidate_tokens.append(problem_path.stem.lower())
        candidate_tokens = [tok for tok in candidate_tokens if tok]

        for token in candidate_tokens:
            for i, problem in enumerate(problems):
                sample_id = str(problem.get("sample_id", "")).lower()
                problem_name = str(problem.get("problem_name", "")).lower()
                if token in sample_id or token in problem_name:
                    return i

        for i, problem in enumerate(problems):
            if problem.get("domain_name", "").lower() == self._ssr_domain_name:
                return i
        return 0

    def __del__(self) -> None:
        super().__del__()
        if hasattr(self, "_image_dir") and os.path.exists(self._image_dir):
            shutil.rmtree(self._image_dir, ignore_errors=True)
        if getattr(self, "_kitchen_renderer", None) is not None:
            try:
                self._kitchen_renderer.close()
            except Exception:
                pass

    def _build_kitchen_renderer(self) -> None:
        module_path = _REPO_ROOT / "experiments" / "final_clean" / "domains" / "builders" / "generate_kitchen_data.py"
        spec = importlib.util.spec_from_file_location("kitchen_data_builder", str(module_path))
        kitchen_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(kitchen_mod)
        kitchen_mod._setup_pybullet_env()
        scene_dir = str(kitchen_mod.KW / "test_cases" / "test_pr2_kitchen")
        self._kitchen_scene_dir = scene_dir
        self._kitchen_renderer = kitchen_mod.KitchenRenderer(scene_dir)
        self._kitchen_module = kitchen_mod

    def _ensure_kitchen_renderer(self) -> None:
        if self._ssr_domain_name not in {"kitchen_worlds", "cooking"}:
            return
        renderer = self._kitchen_renderer
        module = self._kitchen_module
        needs_rebuild = renderer is None or module is None
        if not needs_rebuild:
            try:
                if not bool(renderer._p.isConnected()):  # type: ignore[attr-defined]
                    needs_rebuild = True
            except Exception:
                needs_rebuild = True
        if not needs_rebuild:
            return
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        self._kitchen_renderer = None
        self._kitchen_module = None
        self._build_kitchen_renderer()

    def reset(self) -> State:
        state = super().reset()
        self._object_call_count = 0
        self._object_presence_counts.clear()
        self._object_canonical.clear()
        self._last_objects_instance = None
        self._stable_objects_instance = None
        self._stable_object_keys.clear()
        self._state_observation_count = 0
        return state

    @staticmethod
    def _entity_name(entity: Any) -> str:
        name = getattr(entity, "name", None)
        if name:
            return str(name)
        raw = str(entity)
        return raw.split(":", 1)[0]

    def _build_pybullet_hanoi_state(self, state: State) -> tuple[list[str], list[str], dict[str, Any]]:
        pegs: set[str] = set()
        state_on: dict[str, str] = {}
        raw_clear: dict[str, bool] = {}
        smaller_pairs: list[tuple[str, str]] = []

        for lit in state.literals:
            pred = lit.predicate.name.lower()
            args = [self._entity_name(v) for v in lit.variables]
            if pred == "on" and len(args) == 2:
                state_on[args[0]] = args[1]
                if args[1].lower().startswith("peg"):
                    pegs.add(args[1])
            elif pred == "clear" and len(args) == 1:
                raw_clear[args[0]] = True
                if args[0].lower().startswith("peg"):
                    pegs.add(args[0])
            elif pred == "smaller" and len(args) == 2:
                smaller_pairs.append((args[0], args[1]))

        if self.state is not None:
            for obj in self.state.objects:
                obj_name = self._entity_name(obj)
                if obj_name.lower().startswith("peg"):
                    pegs.add(obj_name)

        pegs_list = sorted(pegs)
        discs = sorted([d for d in state_on.keys() if d not in pegs_list])
        if not pegs_list:
            raise ValueError("Could not infer hanoi pegs from current state.")
        if not discs:
            raise ValueError("Could not infer active hanoi discs from current state.")

        state_clear = {d: raw_clear.get(d, False) for d in discs}
        if smaller_pairs:
            smaller_count = {d: 0 for d in discs}
            for larger, smaller in smaller_pairs:
                if larger in smaller_count and smaller in smaller_count:
                    smaller_count[larger] += 1
            discs = sorted(discs, key=lambda d: (smaller_count.get(d, 0), d))

        return pegs_list, discs, {"on": state_on, "clear": state_clear}

    def _render_pybullet_hanoi_state_to_image(self, state: State, image_path: str) -> None:
        _pybullet_env_setup()
        from test_hanoi import HanoiSimulator

        pegs, discs, user_state = self._build_pybullet_hanoi_state(state)
        sim = HanoiSimulator(
            name_disks=discs,
            peg_names=pegs,
            use_gui=False,
            random_seed=42,
            initial_state=user_state,
        )
        try:
            sim.make_observation(sim.state, filename=str(image_path))
        finally:
            sim.shutdown()

    def _render_state_to_image(self, state: State) -> str:
        if self._ssr_domain_name in {"kitchen_worlds", "cooking"}:
            self._ensure_kitchen_renderer()
            if self._kitchen_renderer is None or self._kitchen_module is None:
                raise RuntimeError("No kitchen renderer available for VLM observations.")
        elif self._pybullet_hanoi_renderer:
            pass
        elif self._render_fn is None:
            raise RuntimeError(f"No PDDLGym renderer available for domain '{self._ssr_domain_name}'.")

        literal_str = str(sorted(str(l) for l in state.literals))
        state_hash = hashlib.md5(literal_str.encode()).hexdigest()[:16]
        image_path = os.path.join(self._image_dir, f"obs_{state_hash}.png")

        if not os.path.exists(image_path):
            if self._ssr_domain_name in {"kitchen_worlds", "cooking"}:
                literal_atoms = set()
                robot_loc = None
                holding_item = None
                for lit in state.literals:
                    s = str(lit)
                    if "(" in s and s.endswith(")"):
                        pred = s[: s.index("(")]
                        inner = s[s.index("(") + 1 : -1]
                        args = []
                        for token in inner.split(","):
                            token = token.strip()
                            if ":" in token:
                                token = token.split(":", 1)[0]
                            if token:
                                args.append(token)
                        atom = " ".join([pred] + args)
                    else:
                        atom = s
                    literal_atoms.add(atom)
                    if atom.startswith("at-robot robot "):
                        try:
                            robot_loc = atom.split()[2]
                        except Exception:
                            robot_loc = None
                    elif atom.startswith("holding robot "):
                        try:
                            holding_item = atom.split()[2]
                        except Exception:
                            holding_item = None

                camera_name = "overview"
                if robot_loc is not None:
                    camera_name = self._kitchen_module._camera_for_location(robot_loc, fallback="overview")

                try:
                    self._kitchen_renderer.set_pddl_state(literal_atoms)
                except Exception as exc:
                    if "physics server" in str(exc).lower() or "not connected" in str(exc).lower():
                        log.warning("VLMPddlFileSimulator: kitchen renderer lost connection during set_pddl_state, rebuilding")
                        self._ensure_kitchen_renderer()
                        if self._kitchen_renderer is None:
                            raise RuntimeError("No kitchen renderer available for VLM observations.") from exc
                        self._kitchen_renderer.set_pddl_state(literal_atoms)
                    else:
                        raise
                main_img = self._kitchen_renderer.render(
                    self._kitchen_module.CAMERAS[camera_name], width=1600, height=1200
                )
                if self._kitchen_show_robot_panel:
                    side_views = []
                    if camera_name != "overview":
                        overview_img = self._kitchen_renderer.render(
                            self._kitchen_module.CAMERAS["overview"], width=1280, height=960
                        )
                        overview_img = self._kitchen_module.annotate_kitchen_overview(
                            overview_img, robot_loc=robot_loc, held_item=holding_item
                        )
                        side_views.append(overview_img)
                    if "top_down" in self._kitchen_module.CAMERAS:
                        top_img = self._kitchen_renderer.render(
                            self._kitchen_module.CAMERAS["top_down"], width=1280, height=960
                        )
                        top_img = self._kitchen_module.annotate_kitchen_topdown(
                            top_img, robot_loc=robot_loc, held_item=holding_item
                        )
                        side_views.append(top_img)
                    ego_cam = self._kitchen_renderer.ego_camera(robot_loc)
                    if ego_cam:
                        robot_img = self._kitchen_renderer.render(ego_cam, width=1280, height=960)
                        side_views.append(robot_img)
                    head_cam = self._kitchen_renderer.head_task_camera(robot_loc)
                    if head_cam:
                        head_img = self._kitchen_renderer.render(head_cam, width=1280, height=960)
                        side_views.append(head_img)
                    manip_cam = self._kitchen_renderer.manipulation_camera(robot_loc)
                    if manip_cam:
                        manip_img = self._kitchen_renderer.render(manip_cam, width=1280, height=960)
                        side_views.append(manip_img)
                    side_cam = self._kitchen_renderer.held_object_side_camera(robot_loc)
                    top_hold_cam = self._kitchen_renderer.held_object_top_camera(robot_loc)
                    grip_cam = self._kitchen_renderer.held_object_grip_camera()
                    held_body_name = None
                    if holding_item is not None:
                        held_body_name = self._kitchen_module.ITEM_TO_BODY.get(holding_item)
                    if side_cam:
                        if held_body_name:
                            side_img = self._kitchen_renderer.render_with_highlight(
                                side_cam, width=1280, height=960, body_names=[held_body_name]
                            )
                        else:
                            side_img = self._kitchen_renderer.render(side_cam, width=1280, height=960)
                        side_views.append(side_img)
                    if top_hold_cam:
                        if held_body_name:
                            top_hold_img = self._kitchen_renderer.render_with_highlight(
                                top_hold_cam, width=1280, height=960, body_names=[held_body_name]
                            )
                        else:
                            top_hold_img = self._kitchen_renderer.render(top_hold_cam, width=1280, height=960)
                        side_views.append(top_hold_img)
                    if grip_cam:
                        if held_body_name:
                            grip_img = self._kitchen_renderer.render_with_highlight(
                                grip_cam, width=1280, height=960, body_names=[held_body_name]
                            )
                        else:
                            grip_img = self._kitchen_renderer.render(grip_cam, width=1280, height=960)
                        side_views.append(grip_img)
                    if side_views:
                        img_data = _concat_images_horizontally([main_img] + side_views)
                    else:
                        img_data = main_img
                else:
                    img_data = main_img
                imageio.imwrite(image_path, np.array(img_data))
            elif self._pybullet_hanoi_renderer:
                self._render_pybullet_hanoi_state_to_image(state, image_path)
            else:
                img_data = self._render_fn(state.literals)
                plt.close("all")
                if isinstance(img_data, np.ndarray):
                    img_data = Image.fromarray(img_data)
                if img_data.mode == "RGBA":
                    img_data = img_data.convert("RGB")
                imageio.imwrite(image_path, np.array(img_data))

        return image_path

    @staticmethod
    def _state_hash(state: State) -> str:
        literal_str = str(sorted(str(l) for l in state.literals))
        return hashlib.md5(literal_str.encode()).hexdigest()[:16]

    def _cache_file(self, state_hash: str) -> Path:
        return self._cache_dir / f"{state_hash}.pkl"

    def _load_cached_observation(self, state_hash: str) -> Optional[Dict[str, Any]]:
        path = self._cache_file(state_hash)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                cached = pickle.load(f)
            if not isinstance(cached, dict):
                return None
            return cached
        except Exception:
            return None

    def _save_cached_observation(
        self,
        state_hash: str,
        observed_atoms: Set[Literal],
        atom_probabilities: Dict[Literal, float],
        last_logprobs: Optional[Dict[str, Any]],
    ) -> None:
        path = self._cache_file(state_hash)
        payload = {
            "observed_atoms": observed_atoms,
            "atom_probabilities": atom_probabilities,
            "last_logprobs": last_logprobs,
        }
        try:
            with path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    @staticmethod
    def _extract_object_records(objects_instance: Any) -> list[Dict[str, str]]:
        records: list[Dict[str, str]] = []
        for obj in getattr(objects_instance, "objects", []):
            name = getattr(obj, "name", None)
            obj_type = getattr(obj, "type", None)
            if not name or not obj_type:
                continue
            records.append({"name": str(name), "type": str(obj_type)})
        return records

    def _build_objects_instance(self, template_instance: Any, records: list[Dict[str, str]]) -> Any:
        cls = type(template_instance)
        payload = {"objects": records}
        if hasattr(cls, "model_validate"):
            return cls.model_validate(payload)
        if hasattr(cls, "parse_obj"):
            return cls.parse_obj(payload)
        return template_instance

    def _should_refresh_objects(self) -> bool:
        if self._last_objects_instance is None:
            return True
        idx = self._state_observation_count
        if idx <= self._object_refresh_first_n:
            return True
        if self._object_refresh_period > 0 and ((idx - self._object_refresh_first_n) % self._object_refresh_period == 0):
            return True
        return False

    def _update_object_profile(self, objects_instance: Any) -> None:
        records = self._extract_object_records(objects_instance)
        if not records:
            return
        self._object_call_count += 1
        for rec in records:
            key = (rec["name"], rec["type"])
            self._object_presence_counts[key] = self._object_presence_counts.get(key, 0) + 1
            self._object_canonical[key] = rec
        self._last_objects_instance = objects_instance
        if self._object_call_count <= 0:
            self._stable_objects_instance = objects_instance
            return
        if self._accumulate_objects:
            merged = [self._object_canonical[k] for k in sorted(self._object_canonical)]
            self._stable_object_keys = set(self._object_canonical.keys())
            self._stable_objects_instance = self._build_objects_instance(objects_instance, merged)
            return
        stable = []
        stable_keys: Set[tuple[str, str]] = set()
        for key, count in self._object_presence_counts.items():
            if (count / self._object_call_count) >= self._object_stability_threshold:
                stable_keys.add(key)
                stable.append(self._object_canonical[key])
        if not stable:
            stable = records
            stable_keys = {(r["name"], r["type"]) for r in records}
        self._stable_object_keys = stable_keys
        self._stable_objects_instance = self._build_objects_instance(objects_instance, stable)

    def _get_objects_for_predicates(self) -> Any:
        if self._stable_objects_instance is not None:
            return self._stable_objects_instance
        return self._last_objects_instance

    def _literal_from_ssr(self, predicate_name: str, raw_args: list, state: State) -> Optional[Literal]:
        fallback_type = self._infer_fallback_object_type(state)
        gt_map: Dict[str, TypedEntity] = {}
        if state and state.objects:
            for obj in state.objects:
                gt_map[obj.name] = obj

        typed_args = []
        for raw in raw_args:
            if raw is None:
                continue
            if isinstance(raw, TypedEntity):
                typed_args.append(gt_map.get(raw.name, raw))
                continue
            name = getattr(raw, "name", str(raw))
            if name in gt_map:
                typed_args.append(gt_map[name])
            else:
                typed_args.append(TypedEntity.__new__(TypedEntity, name, fallback_type))

        norm = predicate_name.strip().lower()
        for candidate in (norm, norm.replace("_", "-"), norm.replace("-", "_"), norm.replace("-", ""), norm.replace("_", "")):
            if candidate in self.domain.predicates:
                pred = self.domain.predicates[candidate]
                return Literal(pred, typed_args)

        compact = norm.replace("-", "").replace("_", "")
        for dname, pred in self.domain.predicates.items():
            if compact == dname.replace("-", "").replace("_", ""):
                return Literal(pred, typed_args)

        log.debug("VLM obs: unknown predicate '%s' – skipped", predicate_name)
        return None

    @staticmethod
    def _infer_fallback_object_type(state: Optional[State]) -> Type:
        if state and state.objects:
            present_types = {obj.var_type for obj in state.objects if isinstance(getattr(obj, "var_type", None), Type)}
            for obj_type in present_types:
                if str(obj_type) == "default":
                    return obj_type
            if len(present_types) == 1:
                return next(iter(present_types))
        return Type("default")

    def make_observation(self, state: State) -> Dict[str, Set[Literal]]:
        state_hash = self._state_hash(state)
        cached = self._load_cached_observation(state_hash)
        if cached is not None:
            self._atom_probabilities = dict(cached.get("atom_probabilities", {}))
            self._last_logprobs = cached.get("last_logprobs")
            observed_atoms = set(cached.get("observed_atoms", set()))
            self._last_usage = {
                "objects": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "predicates": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "total": {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            self._usage_totals["cache_hits"] += 1
            return {"atoms": observed_atoms}

        try:
            image_path = self._render_state_to_image(state)
        except RuntimeError as exc:
            raise RuntimeError(f"VLM render failed: {exc}") from exc

        self._state_observation_count += 1
        obj_usage: Dict[str, Any] = {}
        should_refresh = self._should_refresh_objects()
        retrieved_objs = None
        if should_refresh:
            try:
                retrieved_objs_list, _ = retrieve_objects(
                    image_path=image_path,
                    domain_name=self._ssr_domain_name,
                    dataset=self._dataset,
                    num_examples=self._num_examples,
                    dataset_path=self._dataset_path,
                    number_answers=1,
                    problem_id=self._problem_id,
                    _out_usage=obj_usage,
                )
            except Exception as exc:
                # If refresh fails, fall back to reusable objects when available.
                fallback_objs = self._get_objects_for_predicates()
                if fallback_objs is None:
                    raise RuntimeError(f"VLM retrieve_objects failed: {exc}") from exc
                retrieved_objs = fallback_objs
                obj_usage = {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            else:
                if not retrieved_objs_list or retrieved_objs_list[0] is None:
                    fallback_objs = self._get_objects_for_predicates()
                    if fallback_objs is None:
                        raise RuntimeError("VLM retrieve_objects returned empty result")
                    retrieved_objs = fallback_objs
                    obj_usage = {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                else:
                    retrieved_objs = retrieved_objs_list[0]
                    self._update_object_profile(retrieved_objs)
                    retrieved_objs = self._get_objects_for_predicates()
                    self._usage_totals["calls_objects"] += 1
        else:
            retrieved_objs = self._get_objects_for_predicates()
            if retrieved_objs is None:
                # Safety fallback if no object profile exists yet.
                try:
                    retrieved_objs_list, _ = retrieve_objects(
                        image_path=image_path,
                        domain_name=self._ssr_domain_name,
                        dataset=self._dataset,
                        num_examples=self._num_examples,
                        dataset_path=self._dataset_path,
                        number_answers=1,
                        problem_id=self._problem_id,
                        _out_usage=obj_usage,
                    )
                except Exception as exc:
                    raise RuntimeError(f"VLM retrieve_objects failed: {exc}") from exc
                if not retrieved_objs_list or retrieved_objs_list[0] is None:
                    raise RuntimeError("VLM retrieve_objects returned empty result")
                retrieved_objs = retrieved_objs_list[0]
                self._update_object_profile(retrieved_objs)
                retrieved_objs = self._get_objects_for_predicates()
                self._usage_totals["calls_objects"] += 1
            else:
                self._usage_totals["objects_reused"] += 1
                obj_usage = {"available": False, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        raw_logprobs: Dict = {}
        pred_usage: Dict[str, Any] = {}
        try:
            retrieved_preds_list, pred_probs_list = retrieve_predicates(
                image_path=image_path,
                domain_name=self._ssr_domain_name,
                objects=retrieved_objs,
                dataset=self._dataset,
                num_examples=self._num_examples,
                dataset_path=self._dataset_path,
                number_answers=1,
                problem_id=self._problem_id,
                _out_logprobs=raw_logprobs,
                _out_usage=pred_usage,
            )
        except Exception as exc:
            raise RuntimeError(f"VLM retrieve_predicates failed: {exc}") from exc
        self._last_logprobs = raw_logprobs if raw_logprobs else None
        obj_prompt = int(obj_usage.get("prompt_tokens", 0) or 0)
        obj_completion = int(obj_usage.get("completion_tokens", 0) or 0)
        obj_total = int(obj_usage.get("total_tokens", 0) or 0)
        pred_prompt = int(pred_usage.get("prompt_tokens", 0) or 0)
        pred_completion = int(pred_usage.get("completion_tokens", 0) or 0)
        pred_total = int(pred_usage.get("total_tokens", 0) or 0)
        total_prompt = obj_prompt + pred_prompt
        total_completion = obj_completion + pred_completion
        total_tokens = obj_total + pred_total
        any_usage = bool(obj_usage.get("available", False) or pred_usage.get("available", False))
        self._last_usage = {
            "objects": {
                "available": bool(obj_usage.get("available", False)),
                "prompt_tokens": obj_prompt,
                "completion_tokens": obj_completion,
                "total_tokens": obj_total,
            },
            "predicates": {
                "available": bool(pred_usage.get("available", False)),
                "prompt_tokens": pred_prompt,
                "completion_tokens": pred_completion,
                "total_tokens": pred_total,
            },
            "total": {
                "available": any_usage,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
            },
        }
        self._usage_totals["prompt_tokens"] += total_prompt
        self._usage_totals["completion_tokens"] += total_completion
        self._usage_totals["total_tokens"] += total_tokens
        self._usage_totals["calls_predicates"] += 1

        if not retrieved_preds_list or retrieved_preds_list[0] is None:
            raise RuntimeError("VLM retrieve_predicates returned empty result")

        retrieved_preds = retrieved_preds_list[0]
        all_pred_probs = pred_probs_list[0] if pred_probs_list else []

        observed_atoms: Set[Literal] = set()
        raw_preds = list(getattr(retrieved_preds, "grounded_predicates", []) or [])
        dropped_predicates = 0
        for gp in raw_preds:
            args = [getattr(gp, a) for a in ("x", "y", "z") if getattr(gp, a, None) is not None]
            lit = self._literal_from_ssr(gp.predicate_type, args, state)
            if lit is not None:
                observed_atoms.add(lit)
            else:
                dropped_predicates += 1

        if dropped_predicates > 0:
            log.warning(
                "VLM predicate mapping dropped %d/%d predictions for state hash=%s",
                dropped_predicates,
                len(raw_preds),
                state_hash[:12],
            )
        if raw_preds and not observed_atoms:
            log.warning(
                "All predicted predicates were filtered out for state hash=%s (raw=%d).",
                state_hash[:12],
                len(raw_preds),
            )

        self._atom_probabilities = {lit: 1.0 for lit in observed_atoms}
        for entry in all_pred_probs:
            predic = entry["predicate"]
            prob = entry["prob"]
            args = [getattr(predic, a) for a in ("x", "y", "z") if getattr(predic, a, None) is not None]
            lit = self._literal_from_ssr(predic.predicate_type, args, state)
            if lit is not None:
                self._atom_probabilities[lit] = round(prob, 4)

        self._save_cached_observation(
            state_hash=state_hash,
            observed_atoms=observed_atoms,
            atom_probabilities=self._atom_probabilities,
            last_logprobs=self._last_logprobs,
        )

        return {"atoms": observed_atoms}

    def get_state_probability(self, state: State) -> Dict[Literal, float]:
        return self._atom_probabilities

    def get_last_logprobs(self) -> Optional[Dict]:
        """Return raw logprobs from the most recent VLM observation call."""
        return self._last_logprobs

    def get_last_usage(self) -> Dict[str, Any]:
        """Return token usage for the most recent VLM observation (objects + predicates)."""
        return self._last_usage

    def get_usage_totals(self) -> Dict[str, int]:
        """Return cumulative token usage over all VLM observation calls for this simulator."""
        return dict(self._usage_totals)
