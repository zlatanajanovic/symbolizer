"""
This script processes a dataset of planning problems and their associated states (images and ground-truth objects/predicates),
and evaluates the performance of object and predicate extraction from images using multiple parsing attempts. 
It uses OpenAI to parse scene elements from images and then evaluates various scenarios by aggregating predictions 
across multiple answers. The evaluations are done for objects and predicates under different scenarios:
- Threshold-based filtering of predictions based on their aggregated probabilities.
- Using each individual answer as a separate trial.
- Frequency-based filtering (e.g., selecting objects/predicates that appear in multiple answers).

Finally, it computes evaluation metrics for objects, predicates, and optionally goals.

Key steps:
1. Load configuration and dataset.
2. For each sampled state in the dataset, retrieve multiple object and predicate predictions from OpenAI.
3. Aggregate predictions across multiple answers to form threshold and frequency-based scenarios.
4. Evaluate predictions against ground truth under different scenarios.
5. Save all intermediate results and evaluation metrics.
"""

import argparse
import os
import json
import random
import sys
import time
from typing import List, Dict, Any

# External dependencies
from dotenv import load_dotenv
from pydantic import ValidationError

# Patch: allow 'object' and 'type' as PDDL type names.
# The pddl library overly restricts them as keywords, breaking kitchen-worlds domains.
import pddl.custom_types as _pddl_ct
_pddl_ct.ALL_SYMBOLS.discard("object")
_pddl_ct.ALL_SYMBOLS.discard("type")

# Patch: include 'object' as a valid type in the pddl type hierarchy.
# PDDL treats 'object' as the implicit root type, but the pddl library's
# validator doesn't include it in the available types set when checking
# parameter types like '?o1 - object'.
import pddl._validation as _pddl_val
_orig_get_all_types = _pddl_val.Types._get_all_types
def _patched_get_all_types(self):
    result = _orig_get_all_types(self)
    result.add("object")
    return result
_pddl_val.Types._get_all_types = _patched_get_all_types

# Parse command line arguments early
def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on dataset')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to config file (default: ./config.yaml)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Override dataset_path from config')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Override output_folder from config')
    parser.add_argument('--domain_name', type=str, default=None,
                        help='Override domain_name from config')
    parser.add_argument('--num_tests', type=int, default=None,
                        help='Override num_tests from config')
    parser.add_argument('--oracle_stage', type=str, default=os.getenv("SSR_ORACLE_STAGE", "none"),
                        choices=["none", "objects", "objects_predicates"],
                        help='Cascade oracle stage: none, objects, or objects_predicates')
    parser.add_argument('--dry-run', action='store_true',
                        help='Enumerate sampled work and exit before any API calls')
    return parser.parse_args()

_args = parse_args()
CONFIG_PATH = _args.config

# Domain and parsing utilities
from .parsing_utils.general_utils import load_env, load_config
from .parsing_utils.generate_scenes_utils import get_atom_schema_from_chatgpt_objects
from .parsing_utils.chatgpt_utils import get_object_schema_from_pddl
from .run_eval_vlm import (
    load_dataset, 
    evaluate_predicates, 
    evaluate_objects, 
    calculate_metrics, 
    calculate_metrics_objects,
    reinitialize_client
)
from .ssr_parsing import retrieve_objects, retrieve_predicates, get_simplified_object_probs, get_simplified_pred_probs, retrieve_goal

# OpenAI wrapper
from openai import OpenAI
from mistralai import Mistral
from pydantic import BaseModel, Field
from typing import Literal


def _rebuild_object_schema(all_types):
    """Rebuild object schema with an expanded set of types."""
    list_types = tuple(sorted(all_types))
    class Object(BaseModel):
        name: str
        type: Literal[list_types] = Field(..., description="Must be one of the valid types")
    class ObjectList(BaseModel):
        objects: List[Object]
    return ObjectList, list_types


def _extract_atom_object_names(atoms_json_str: str) -> set:
    """Extract all object names referenced in atoms JSON."""
    atoms_data = json.loads(atoms_json_str)
    names = set()
    for pred in atoms_data.get("grounded_predicates", []):
        for key, val in pred.items():
            if isinstance(val, dict) and "name" in val:
                names.add(val["name"])
    return names


def _augment_gt_objects_for_atoms(gt_objects_instance, atoms_json_str: str, domain_file: str):
    """
    Augment GT objects with any extra objects referenced in atoms but not in
    the GT objects list. Returns a new object instance with the augmented list.
    """
    atom_names = _extract_atom_object_names(atoms_json_str)
    existing_names = {o.name for o in gt_objects_instance.objects}
    extra_names = atom_names - existing_names
    if not extra_names:
        return gt_objects_instance

    # Build augmented objects list — assign 'object' type to extras
    all_objs = [{"name": o.name, "type": str(o.type)} for o in gt_objects_instance.objects]
    for name in sorted(extra_names):
        all_objs.append({"name": name, "type": "object"})

    all_types = {o["type"] for o in all_objs}
    schema, _ = _rebuild_object_schema(tuple(sorted(all_types)))
    return schema(**{"objects": all_objs})


def _build_permissive_atoms_schema(gt_objects_instance, atoms_json_str: str, domain_file: str):
    """
    Build a permissive Atoms schema for GT parsing where every predicate
    parameter type class includes ALL object names. This handles GT data
    where objects appear in predicate positions that don't match their PDDL types.
    """
    # Collect ALL object names (from objects + atoms)
    all_names = {o.name for o in gt_objects_instance.objects}
    all_names |= _extract_atom_object_names(atoms_json_str)
    all_names_list = sorted(all_names)

    from .parsing_utils.generate_scenes_utils import PDDLDomainWrapper, next_letter, parse_domain
    from .parsing_utils.chatgpt_utils import create_dynamic_models
    from pydantic import create_model

    # Build type mapping where EVERY type has ALL object names
    domain = parse_domain(domain_file)
    domain_types = {}
    for key, value in domain.types.items():
        if value not in domain_types:
            domain_types[value] = []
        domain_types[value].append(key)

    types_available_classes_mapping = {}
    for obj in gt_objects_instance.objects:
        t = str(obj.type)
        if t not in types_available_classes_mapping:
            types_available_classes_mapping[t] = list(all_names_list)
        
    # Ensure supertype classes also have all names
    for superclass in domain_types.keys():
        if superclass and "None" not in superclass:
            types_available_classes_mapping[superclass] = list(all_names_list)
    
    # Ensure 'object' root type has all names
    types_available_classes_mapping["object"] = list(all_names_list)

    domain_wrapper = PDDLDomainWrapper(domain_file)
    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}

    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
            # Ensure the type is in the mapping
            if type_name not in types_available_classes_mapping:
                types_available_classes_mapping[type_name] = list(all_names_list)
        predicates_structured[pred_name] = dict_how_are_they_named

    _, Atoms, _ = create_dynamic_models(predicates_structured, types_available_classes_mapping)
    return Atoms


# Load configuration - use CONFIG_PATH from command line argument
print(f"Loading config from: {CONFIG_PATH}")
config = load_config(CONFIG_PATH)
if _args.dataset_path or os.getenv("SYMBOLIZER_DATASET_OVERRIDE"):
    config["dataset_path"] = _args.dataset_path or os.getenv("SYMBOLIZER_DATASET_OVERRIDE")
if _args.output_folder or os.getenv("SYMBOLIZER_OUTPUT_OVERRIDE"):
    config["output_folder"] = _args.output_folder or os.getenv("SYMBOLIZER_OUTPUT_OVERRIDE")
if _args.domain_name or os.getenv("SYMBOLIZER_DOMAIN_OVERRIDE"):
    config["domain_name"] = _args.domain_name or os.getenv("SYMBOLIZER_DOMAIN_OVERRIDE")
if _args.num_tests is not None or os.getenv("SYMBOLIZER_NUM_TESTS_OVERRIDE"):
    config["num_tests"] = int(_args.num_tests if _args.num_tests is not None else os.getenv("SYMBOLIZER_NUM_TESTS_OVERRIDE"))
load_env(config['env_file'])

# Reinitialize the VLM client with the new environment settings
if _args.dry_run:
    print("[DRY RUN] Skipping VLM client reinitialization.")
else:
    reinitialize_client()

# Safety check: verify model matches config expectation
_active_model = os.getenv("MODEL_TO_USE", "unknown")
_active_name = os.getenv("MODEL_NAME", "unknown")
_env_file = config.get('env_file', 'unknown')
print(f"[MODEL GUARD] Config={CONFIG_PATH}, env_file={_env_file}, "
      f"MODEL_TO_USE={_active_model}, MODEL_NAME={_active_name}")
# Detect mismatch: if config path contains a model hint, verify it
for _hint, _expected in [("gpt52", "openai"), ("mistral", "mistral"), ("gemini", "gemini")]:
    if _hint in CONFIG_PATH.lower() and _active_model != _expected:
        print(f"[MODEL GUARD] WARNING: Config path contains '{_hint}' but "
              f"MODEL_TO_USE={_active_model}. Possible model/config mismatch!")

# OpenAI API settings
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
MODEL_NAME = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_TO_USE = os.getenv("MODEL_TO_USE", "None Set")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "pixtral-large-latest")

# Configuration Parameters
MAX_TOKENS = config['max_tokens']
DETAIL_LEVEL_LOW = config['detail_level_low']
DATASET_PATH = config['dataset_path']
NUM_EXAMPLES = config['num_examples']
NUM_SAMPLES = config['num_tests']
DOMAIN_NAME = config['domain_name']
OUTPUT_FOLDER = config['output_folder']
VERBOSE = config['verbose']
FULL_RESPONSES_OBJECTS_FILE = config["objects_full_responses_path"]
FULL_RESPONSES_ATOMS_FILE = config["atoms_full_responses_path"]
FULL_RESPONSES_GOALS_FILE = config["goals_full_responses_path"]
NUM_ANSWERS_PER_SAMPLE = config['num_answers_per_sample']
ORACLE_STAGE = _args.oracle_stage
RESUME_RUN = os.getenv("SSR_RESUME", "false").lower() == "true"
USE_DIRECT_OBJECTS = os.getenv("SSR_USE_DIRECT_OBJECTS", "false").lower() == "true"
INCLUDE_INSTRUCTION_IN_OBJECT_ATOM_PROMPTS = (
    os.getenv("SSR_INCLUDE_INSTRUCTION_IN_OBJECT_ATOM_PROMPTS", "false").lower() == "true"
)
_seed_env = os.getenv("SSR_RANDOM_SEED", "").strip()
if _seed_env:
    try:
        _seed_value = int(_seed_env)
        random.seed(_seed_value)
        print(f"[SEED] SSR_RANDOM_SEED={_seed_value}")
    except ValueError:
        print(f"[SEED] Invalid SSR_RANDOM_SEED={_seed_env!r}; ignoring")

# Threshold values for evaluation scenarios
# THRESHOLDS = [0, 0.05, 0.1, 0.2, 0.3]
THRESHOLDS = [0.3]
# Keep original behavior by default (no forced GT objects).
# Optional override: SSR_USE_GT_OBJECTS=blocks,hanoi
_gt_obj_env = os.getenv("SSR_USE_GT_OBJECTS", "").strip()
USE_GROUND_TRUTH_OBJ = [d.strip() for d in _gt_obj_env.split(",") if d.strip()] if _gt_obj_env else []

def aggregate_object_probabilities(objects_probs_list: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate object probabilities from multiple answers.
    Each element in objects_probs_list corresponds to one answer (a list of objects with probabilities).
    Returns a dictionary keyed by object name with aggregated probability statistics.
    """
    dict_object_probs = {}
    for object_probs_element in objects_probs_list:
        simpl_obj_probs = get_simplified_object_probs(object_probs_element) 
        for simple_obj in simpl_obj_probs:
            obj_name = str(simple_obj["object"])
            if obj_name in dict_object_probs:
                dict_object_probs[obj_name]["num_times"] += 1
                dict_object_probs[obj_name]["probability_sum"] += simple_obj["prob"]
            else:
                dict_object_probs[obj_name] = {
                    "num_times": 1, 
                    "probability_sum": simple_obj["prob"], 
                    "object": simple_obj["object"]
                }
    # Compute average probability
    for key in dict_object_probs:
        dict_object_probs[key]["probability"] = dict_object_probs[key]["probability_sum"] / dict_object_probs[key]["num_times"]
    return dict_object_probs

def aggregate_predicate_probabilities(predicates_probs_list: List[List[Dict[str, Any]]],
                                    simplify:bool=True) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate predicate probabilities from multiple answers.
    Each element in predicates_probs_list is a list of predicates with associated probabilities.
    Returns a dictionary keyed by predicate string.
    """
    dict_pred_probs = {}
    for pred_probs_element_1 in predicates_probs_list:
        if not pred_probs_element_1:
            continue
        if simplify:
            simpl_pred_probs_element = get_simplified_pred_probs(pred_probs_element_1) 
        else:
            simpl_pred_probs_element = pred_probs_element_1
        if not simpl_pred_probs_element:
            continue
        for pred_obj in simpl_pred_probs_element:
            pred_str = pred_obj["predicate"].__str__()
            if pred_str in dict_pred_probs:
                dict_pred_probs[pred_str]["num_times"] += 1
                dict_pred_probs[pred_str]["probability_sum"] += pred_obj["prob"]
            else:
                dict_pred_probs[pred_str] = {
                    "num_times": 1, 
                    "probability_sum": pred_obj["prob"], 
                    "predicate": pred_obj["predicate"]
                }
    # Compute average probability
    for key in dict_pred_probs:
        dict_pred_probs[key]["probability"] = dict_pred_probs[key]["probability_sum"] / dict_pred_probs[key]["num_times"]
    return dict_pred_probs

def filter_objects_by_threshold(dict_object_probs: Dict[str, Dict[str, Any]], threshold: float) -> List[Any]:
    """
    Returns a list of objects whose average probability is greater than or equal to the given threshold.
    """
    return [vals["object"] for obj, vals in dict_object_probs.items() if vals["probability"] >= threshold]

def filter_predicates_by_threshold(dict_pred_probs: Dict[str, Dict[str, Any]], threshold: float) -> List[Any]:
    """
    Returns a list of predicates whose average probability is greater than or equal to the given threshold.
    """
    return [vals["predicate"] for pred, vals in dict_pred_probs.items() if vals["probability"] >= threshold]

def filter_objects_by_frequency(dict_object_probs: Dict[str, Dict[str, Any]], min_frequency: int = 2) -> List[Any]:
    """
    Returns a list of objects that appear at least 'min_frequency' times across answers.
    """
    return [vals["object"] for obj, vals in dict_object_probs.items() if vals["num_times"] >= min_frequency]

def filter_predicates_by_frequency(dict_pred_probs: Dict[str, Dict[str, Any]], min_frequency: int = 2) -> List[Any]:
    """
    Returns a list of predicates that appear at least 'min_frequency' times across answers.
    """
    return [vals["predicate"] for pred, vals in dict_pred_probs.items() if vals["num_times"] >= min_frequency]

def evaluate_scenarios_objects(
    evaluation_results_objects: List[Dict[str, Any]],
    request_id: str,
    ground_truth_objects: Dict[str, Any],
    objects_instance_list: List[Any],
    dict_object_probs: Dict[str, Dict[str, Any]]
):
    """
    Evaluate objects under the active scenario configuration.
    Currently only:
    1) All answers individually.
    Results are appended to evaluation_results_objects.
    """
    # Scenario 1: threshold-based (disabled by request)
    # for t in THRESHOLDS:
    #     selected_objects = filter_objects_by_threshold(dict_object_probs, t) 
    #     answer_objects = {"objects": [o.dict() for o in selected_objects]}
    #     eval_res = evaluate_objects(answer_objects, ground_truth_objects)
    #     eval_res.update({"request_id": request_id, "scenario": f"threshold_{t}", "parse_index": None})
    #     evaluation_results_objects.append(eval_res)

    # Scenario 2: each answer is a separate trial
    for parse_index, obj_instance in enumerate(objects_instance_list):
        answer_objects = {"objects": obj_instance.dict().get('objects', [])}
        eval_res = evaluate_objects(answer_objects, ground_truth_objects)
        eval_res.update({"request_id": request_id, "scenario": "all_answers_individual", "parse_index": parse_index})
        evaluation_results_objects.append(eval_res)

    # Scenario 3: frequency-based (>2 times) (disabled by request)
    # freq_objects = filter_objects_by_frequency(dict_object_probs, min_frequency=1)
    # answer_objects = {"objects": [o.dict() for o in freq_objects]}
    # eval_res = evaluate_objects(answer_objects, ground_truth_objects)
    # eval_res.update({"request_id": request_id, "scenario": "frequency>1", "parse_index": None})
    # evaluation_results_objects.append(eval_res)

def evaluate_scenarios_predicates(
    evaluation_results_atoms: List[Dict[str, Any]],
    request_id: str,
    ground_truth_atoms: Dict[str, Any],
    predicates_instance_list: List[Any],
    dict_pred_probs: Dict[str, Dict[str, Any]]
):
    """
    Evaluate predicates under the active scenario configuration.
    Currently only:
    1) All answers individually.
    Results are appended to evaluation_results_atoms.
    """
    # Scenario 1: threshold-based (disabled by request)
    # for t in THRESHOLDS:
    #     selected_preds = filter_predicates_by_threshold(dict_pred_probs, t)        
    #     answer_atoms = {"grounded_predicates": [p.dict() for p in selected_preds]}
    #     eval_res = evaluate_predicates(answer_atoms, ground_truth_atoms)
    #     eval_res.update({"request_id": request_id, "scenario": f"threshold_{t}", "parse_index": None})
    #     evaluation_results_atoms.append(eval_res)

    # Scenario 2: each answer individually
    for parse_index, pred_instance in enumerate(predicates_instance_list):
        answer_atoms = {"grounded_predicates": pred_instance.dict().get('grounded_predicates', [])}
        eval_res = evaluate_predicates(answer_atoms, ground_truth_atoms)
        eval_res.update({"request_id": request_id, "scenario": "all_answers_individual", "parse_index": parse_index})
        evaluation_results_atoms.append(eval_res)

    # Scenario 3: frequency-based (>2 times) (disabled by request)
    # freq_preds = filter_predicates_by_frequency(dict_pred_probs, min_frequency=1)
    # answer_atoms = {"grounded_predicates": [p.dict() for p in freq_preds]}
    # eval_res = evaluate_predicates(answer_atoms, ground_truth_atoms)
    # eval_res.update({"request_id": request_id, "scenario": "frequency>1", "parse_index": None})
    # evaluation_results_atoms.append(eval_res)


def aggregate_goal_probabilities(goals_probs_list: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate goal probabilities from multiple answers.
    Each element in goals_probs_list is a list of goal predicates with associated probabilities.
    Returns a dictionary keyed by goal predicate string.
    """
    dict_goal_probs = {}
    for goal_probs_element_1 in goals_probs_list:
        # Here we assume a similar structure as predicates: each goal predicate entry has a 'grounded_predicates' field and 'prob'
        # If different, adjust accordingly.
        for g_obj in goal_probs_element_1:
            g_str = g_obj["predicate"].__str__()
            if g_str in dict_goal_probs:
                dict_goal_probs[g_str]["num_times"] += 1
                dict_goal_probs[g_str]["probability_sum"] += g_obj["prob"]
            else:
                dict_goal_probs[g_str] = {
                    "num_times": 1, 
                    "probability_sum": g_obj["prob"], 
                    "predicate": g_obj["predicate"]
                }
    # Compute average probability
    for key in dict_goal_probs:
        dict_goal_probs[key]["probability"] = dict_goal_probs[key]["probability_sum"] / dict_goal_probs[key]["num_times"]
    return dict_goal_probs

def filter_goals_by_threshold(dict_goal_probs: Dict[str, Dict[str, Any]], threshold: float) -> List[Any]:
    """
    Returns a list of goals (predicates) whose average probability is greater than or equal to the given threshold.
    """
    return [vals["predicate"] for pred, vals in dict_goal_probs.items() if vals["probability"] >= threshold]

def filter_goals_by_frequency(dict_goal_probs: Dict[str, Dict[str, Any]], min_frequency: int = 2) -> List[Any]:
    """
    Returns a list of goals (predicates) that appear at least 'min_frequency' times across answers.
    """
    return [vals["predicate"] for pred, vals in dict_goal_probs.items() if vals["num_times"] >= min_frequency]

def evaluate_scenarios_goals(
    evaluation_results_goals: List[Dict[str, Any]],
    request_id: str,
    ground_truth_goals: Dict[str, Any],
    goals_instance_list: List[Any],
    dict_goal_probs: Dict[str, Dict[str, Any]]
):
    """
    Evaluate goals under the active scenario configuration.
    Currently only:
    1) All answers individually.
    Results are appended to evaluation_results_goals.
    """
    def _goal_entry_to_dict(g):
        if hasattr(g, "dict"):
            return g.dict()
        if isinstance(g, dict):
            return g
        if isinstance(g, list) and len(g) >= 1:
            # Support list form like ["ontop", "a", "b"]
            pred = {"predicate_type": str(g[0]).capitalize()}
            if len(g) >= 2:
                pred["x"] = {"name": g[1]}
            if len(g) >= 3:
                pred["y"] = {"name": g[2]}
            if len(g) >= 4:
                pred["z"] = {"name": g[3]}
            return pred
        return None

    # Scenario 1: threshold-based (disabled by request)
    # for t in THRESHOLDS:
    #     selected_goals = filter_goals_by_threshold(dict_goal_probs, t)        
    #     normalized = [d for d in (_goal_entry_to_dict(g) for g in selected_goals) if d is not None]
    #     answer_goals = {"grounded_predicates": normalized}
    #     eval_res = evaluate_predicates(answer_goals, ground_truth_goals)
    #     eval_res.update({"request_id": request_id, "scenario": f"threshold_{t}", "parse_index": None})
    #     evaluation_results_goals.append(eval_res)

    # Scenario 2: each answer individually
    for parse_index, goal_instance in enumerate(goals_instance_list):
        answer_goals = {"grounded_predicates": goal_instance.dict().get('grounded_predicates', [])}
        eval_res = evaluate_predicates(answer_goals, ground_truth_goals)
        eval_res.update({"request_id": request_id, "scenario": "all_answers_individual", "parse_index": parse_index})
        evaluation_results_goals.append(eval_res)

    # Scenario 3: frequency-based (>2 times) (disabled by request)
    # freq_goals = filter_goals_by_frequency(dict_goal_probs, min_frequency=1)
    # normalized = [d for d in (_goal_entry_to_dict(g) for g in freq_goals) if d is not None]
    # answer_goals = {"grounded_predicates": normalized}
    # eval_res = evaluate_predicates(answer_goals, ground_truth_goals)
    # eval_res.update({"request_id": request_id, "scenario": "frequency>1", "parse_index": None})
    # evaluation_results_goals.append(eval_res)
    
def process_sample(
    dataset: Dict[str, Any],
    dataset_path: str,
    problem_id: int,
    state_num: int,
    num_examples: int,
    id_counter: int,
    evaluation_results_objects: List[Dict[str, Any]],
    evaluation_results_atoms: List[Dict[str, Any]],
    evaluation_results_goals: List[Dict[str, Any]],
    api_key: str = None,
    num_answers: int = 1,
    use_probs: bool = False,
    obs_threshold: float = 0.05,
    use_ground_truth: bool = False,
    oracle_stage: str = "none"
):
    """
    Process a single sample from the dataset by:
    1. Retrieving multiple object parses and evaluating them.
    2. Aggregating object probabilities and evaluating scenarios.
    3. Based on chosen logic, select objects for predicate retrieval.
    4. Retrieve multiple predicate parses and evaluate them under different scenarios.
    5. If goals are included, do a similar evaluation for goals (not currently implemented).

    The results are saved to disk, and evaluation_results_* lists are updated.
    """
    # Extract relevant info from dataset
    problem_json = dataset["problems"][problem_id]
    domain_name = problem_json["domain_name"]
    domain_file = problem_json["domain_file"]
    random_state = problem_json["states"][state_num]

    # Ground-truth information
    atoms_json_curr = random_state["atoms"]
    objects_json_curr_correct = random_state["all_objects"]
    image_path_curr = random_state["image_path"]
    goal_predicates = random_state.get("goal_predicates", None)
    goal_instruction_path = random_state.get("goal_instruction_path", None)
    instruction_text = None
    if (
        INCLUDE_INSTRUCTION_IN_OBJECT_ATOM_PROMPTS
        and goal_instruction_path
        and os.path.exists(goal_instruction_path)
    ):
        with open(goal_instruction_path, "r") as f:
            instruction_text = f.read().strip() or None
    
    # Build schema for objects from domain, augmented with GT object types
    object_schema_curr, list_types_curr = get_object_schema_from_pddl(domain_file)
    gt_objects_data = json.loads(objects_json_curr_correct)
    gt_types = {o["type"] for o in gt_objects_data.get("objects", [])}
    extra_types = gt_types - set(list_types_curr)

    # Validate GT against schema; if it fails (e.g. GT has extra objects not
    # in PDDL :objects), fall back to domain-only schema with expanded types.
    try:
        if extra_types:
            all_types = tuple(sorted(set(list_types_curr) | gt_types))
            object_schema_curr, list_types_curr = _rebuild_object_schema(all_types)
        object_correct_instance_curr = object_schema_curr(**gt_objects_data)
    except Exception as e:
        print(f"[process_sample] GT validation failed ({e}), rebuilding schema from domain types + GT types")
        all_types = tuple(sorted(set(list_types_curr) | gt_types))
        object_schema_curr, list_types_curr = _rebuild_object_schema(all_types)
        object_correct_instance_curr = object_schema_curr(**gt_objects_data)

    request_id = f"{id_counter}_problem_{problem_id}_state_{state_num}_domain_{domain_name}"
    use_gt_objects = (
        use_ground_truth
        or domain_name in USE_GROUND_TRUTH_OBJ
        or oracle_stage in {"objects", "objects_predicates"}
    )
    use_gt_predicates = oracle_stage == "objects_predicates"

    # Retrieve multiple object parses
    if use_gt_objects:
        print(f"[ORACLE] {request_id}: using ground-truth objects; skipping object VLM call.")
        objects_instance_list = [object_correct_instance_curr for _ in range(num_answers)]
        object_probs = [
            {"object": obj, "prob": 1.0, "logprob": 0.0}
            for obj in getattr(object_correct_instance_curr, "objects", [])
        ]
        objects_probs_list = [object_probs for _ in range(num_answers)]
    else:
        objects_instance_list, objects_probs_list = retrieve_objects(
            image_path=image_path_curr,
            domain_name=domain_name,
            dataset=dataset,
            num_examples=num_examples,
            dataset_path=dataset_path,
            number_answers=num_answers,
            problem_id=problem_id,
            instruction_text=instruction_text,
        )
    if not objects_instance_list:
        print(f"Failed to parse Objects response for ID: {request_id}")
        return

    # Save all objects parses
    for parse_index, object_instance_element in enumerate(objects_instance_list):
        parse_suffix = f"_{parse_index}" if len(objects_instance_list) > 1 else ""
        parsed_objects_file = os.path.join(OUTPUT_FOLDER, f"{request_id}{parse_suffix}_objects.json")

        full_obj_resp = {
            "request_id": request_id,
            "objects": object_instance_element.dict().get('objects', []),
            "domain_file": domain_file,
            "oracle_stage": oracle_stage,
        }

        with open(parsed_objects_file, 'w') as f_obj_full:
            json.dump(full_obj_resp, f_obj_full, indent=2)

        with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_OBJECTS_FILE), 'a') as f_obj_global:
            json.dump(full_obj_resp, f_obj_global)
            f_obj_global.write('\n')
    # Aggregate object probabilities and evaluate scenarios
    dict_object_probs = aggregate_object_probabilities(objects_probs_list)
    ground_truth_objects = object_correct_instance_curr.dict()
    evaluate_scenarios_objects(
        evaluation_results_objects,
        request_id,
        ground_truth_objects,
        objects_instance_list,
        dict_object_probs
    )

    # Determine which objects to use for predicate retrieval
    # Optional direct mode: bypass probability aggregation and pass the first parsed
    # Pydantic object list directly to predicate extraction.
    if use_gt_objects:
        objects_for_predicates = object_correct_instance_curr
    elif USE_DIRECT_OBJECTS:
        objects_for_predicates = objects_instance_list[0]
    else:
        if use_probs:
            selected_objects = filter_objects_by_threshold(dict_object_probs, obs_threshold) 
        else:
            num_minimum_obj = int(obs_threshold * num_answers)
            print(f"Minimum number of objects: {num_minimum_obj}")
            selected_objects = filter_objects_by_frequency(dict_object_probs, min_frequency=1)
        answer_objects = {"objects": [o.dict() for o in selected_objects]}        
        objects_for_predicates = object_schema_curr(**answer_objects)
     
    print("Ground truth objects: ", ground_truth_objects)
    print("Selected objects for predicates: ", objects_for_predicates.dict())
    # Retrieve multiple predicate parses
    if use_gt_predicates:
        print(f"[ORACLE] {request_id}: using ground-truth predicates; skipping predicate VLM call.")
        class _RawAtoms:
            def __init__(self, data):
                self._data = data
                self.grounded_predicates = data.get("grounded_predicates", [])
            def dict(self):
                return self._data

        raw_atoms = _RawAtoms(json.loads(atoms_json_curr))
        grounded_atoms_instance_list = [raw_atoms for _ in range(num_answers)]
        predicates_probabilities = [
            [
                {"predicate": pred, "prob": 1.0, "logprob": 0.0}
                for pred in raw_atoms.grounded_predicates
            ]
            for _ in range(num_answers)
        ]
    else:
        grounded_atoms_instance_list, predicates_probabilities = retrieve_predicates(
            image_path=image_path_curr,
            domain_name=domain_name,
            objects=objects_for_predicates,
            dataset=dataset,
            num_examples=num_examples,
            dataset_path=dataset_path,
            number_answers=num_answers,
            problem_id=problem_id,
            instruction_text=instruction_text,
        )
    if not grounded_atoms_instance_list:
        print(f"Warning: No atoms parsed for ID {request_id}. Treating as empty prediction.")
        # Create an empty atoms instance so evaluation still runs
        empty_atoms = type('EmptyAtoms', (), {
            'dict': lambda self: {"grounded_predicates": []},
            'grounded_predicates': [],
        })()
        grounded_atoms_instance_list = [empty_atoms for _ in range(num_answers)]
        predicates_probabilities = [[] for _ in range(num_answers)]
    
    # Save all atoms parses
    for parse_index, grounded_atoms_instance in enumerate(grounded_atoms_instance_list):
        parse_suffix = f"_{parse_index}" if len(grounded_atoms_instance_list) > 1 else ""
        parsed_atoms_file = os.path.join(OUTPUT_FOLDER, f"{request_id}{parse_suffix}_atoms.json")

        full_atoms_resp = {
            "request_id": request_id,
            "grounded_predicates": grounded_atoms_instance.dict().get("grounded_predicates", []),
            "domain_file": domain_file,
            "oracle_stage": oracle_stage,
        }

        with open(parsed_atoms_file, 'w') as f_atoms_full:
            json.dump(full_atoms_resp, f_atoms_full, indent=2)

        with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_ATOMS_FILE), 'a') as f_atoms_global:
            json.dump(full_atoms_resp, f_atoms_global)
            f_atoms_global.write('\n')

    # Evaluate predicates scenarios
    if grounded_atoms_instance_list:
        # Parse GT atoms directly — no schema validation needed for ground truth
        ground_truth_atoms = json.loads(atoms_json_curr)
        dict_pred_probs = aggregate_predicate_probabilities(predicates_probabilities)
        evaluate_scenarios_predicates(
            evaluation_results_atoms,
            request_id,
            ground_truth_atoms,
            grounded_atoms_instance_list,
            dict_pred_probs
        )


    # Goal evaluation (if needed) can be implemented similarly
    ### if the goal instructions exist, then retrieve the goals and evaluate them. Otherwise, return ground truth goals
    # Goal evaluation section
    goal_predicates = random_state.get("goal_predicates", None)
    goal_instruction_path = random_state.get("goal_instruction_path", None)

    use_ground_truth_goal = False
    goals_instance_list = []
    goals_probabilities_list = []

    if goal_predicates:
        # Parse GT goals directly as raw JSON
        ground_truth_goals = json.loads(goal_predicates)
        #check if thr file exists
        if goal_instruction_path and os.path.exists(goal_instruction_path):
            # Attempt to retrieve goals from instructions
            with open(goal_instruction_path, 'r') as f:
                instruction_text = f.read()
                if len(instruction_text.strip()) > 5:
                    # Retrieve multiple goal parses
                    goals_instance_list, goals_probabilities_list = retrieve_goal(
                        instruction_text=instruction_text,
                        domain_name=domain_name,
                        objects=objects_for_predicates,
                        dataset=dataset,
                        num_examples=num_examples,
                        dataset_path=dataset_path,
                        number_answers=num_answers,
                        problem_id=problem_id,
                    )
                else:
                    use_ground_truth_goal = True
        else:
            use_ground_truth_goal = True

        if use_ground_truth_goal:
            # No goal-instruction text available (missing file or <5 chars): we cannot
            # VLM-ground the goal, so the ground-truth goal is emitted as the
            # "prediction". This makes goal F1 ~= 1.0, which is NOT a real grounding
            # score -- warn loudly so it can never be silently mistaken for a live one.
            # (The PDDLGym and ViLaIn `--dataset vilain` sets DO ship instructions, so
            # this only triggers for datasets whose goal instructions were not released,
            # e.g. `--dataset real`.)
            print(
                f"[goal-eval] WARNING: no goal-instruction text for problem "
                f"{problem_id} (domain={domain_name}); emitting GROUND-TRUTH goal as "
                f"the prediction -> goal F1 will be ~1.0 and is NOT a live VLM "
                f"grounding result.",
                flush=True,
            )
            # No predictions retrieved; use ground truth as is
            class _RawGoals:
                """Thin wrapper to emulate Pydantic model for raw GT goal dicts."""
                def __init__(self, data):
                    self._data = data
                    self.grounded_predicates = [
                        type('P', (), {'dict': lambda s, p=p: p})()
                        for p in data.get("grounded_predicates", [])
                    ]
                def dict(self):
                    return self._data
            raw_goal = _RawGoals(ground_truth_goals)
            goals_instance_list = [raw_goal for _ in range(num_answers)]
            goals_probabilities_list = [
                [
                    {"predicate": g, "prob": 1.0, "logprob": 0.0}
                    for g in raw_goal.grounded_predicates
                ]
                for _ in range(num_answers)
            ]

        # Save all goals parses
        for parse_index, goal_instance_element in enumerate(goals_instance_list):
            parse_suffix = f"_{parse_index}" if len(goals_instance_list) > 1 else ""
            parsed_goals_file = os.path.join(OUTPUT_FOLDER, f"{request_id}{parse_suffix}_goals.json")

            full_goals_resp = {
                "request_id": request_id,
                "goal_predicates": goal_instance_element.dict().get("grounded_predicates", []),
                "domain_file": domain_file,
                "oracle_stage": oracle_stage,
            }

            with open(parsed_goals_file, 'w') as f_goals_full:
                json.dump(full_goals_resp, f_goals_full, indent=2)

            with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_GOALS_FILE), 'a') as f_goals_global:
                json.dump(full_goals_resp, f_goals_global)
                f_goals_global.write('\n')
        # Evaluate goals scenarios if we have goal instances
        if goals_instance_list:
            dict_goal_probs = aggregate_predicate_probabilities(goals_probabilities_list,simplify=True)
            evaluate_scenarios_goals(
                evaluation_results_goals,
                request_id,
                ground_truth_goals,
                goals_instance_list,
                dict_goal_probs
            )


    

def create_sequential_calls_from_dataset_for_domain(
    dataset: Dict[str, Any],
    dataset_path: str,
    domain_name: str = "all",
    num_examples: int = 1,
    num_tests: int = 10,
    output_folder: str = None,
    output_filename_objects: str = "objects_evaluation_results.jsonl",
    output_filename_atoms: str = "atoms_evaluation_results.jsonl",
    output_filename_goals: str = "goals_evaluation_results.jsonl",
    api_keys: List[str] = None,
    num_answers_per_sample: int = 1,
    use_probs: bool = False,
    obs_threshold: float = 0.05,
    use_ground_truth: bool = False,
    oracle_stage: str = "none",
    dry_run: bool = False
):
    """
    Orchestrates the entire evaluation process:
    - Selects random states from the dataset.
    - Processes each sampled state by retrieving objects and predicates, evaluating them, and saving results.
    - Writes evaluation results incrementally to files.
    - Computes final metrics after processing all samples.

    Parameters:
    - dataset: The loaded dataset.
    - dataset_path: Path to the dataset.
    - domain_name: Domain to filter by. If "all", process all domains.
    - num_examples: Number of examples to use for prompting.
    - num_tests: Number of states to test.
    - output_folder: Directory to store outputs.
    - api_keys: List of API keys to cycle through if rate-limited.
    - num_answers_per_sample: Number of times to query the model for objects/predicates per sample.
    - use_probs: Boolean indicating if probability-based scenario selection is used.
    - obs_threshold: Threshold for object selection if use_probs is True.
    - use_ground_truth: If True, ground truth objects are used for predicate retrieval instead of parsed objects.
    """
    if output_folder is None:
        output_folder = OUTPUT_FOLDER
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    evaluation_results_objects = []
    evaluation_results_atoms = []
    evaluation_results_goals = []

    # Collect references to all states in the dataset
    # domain_name can be "all", a single name, or comma-separated list
    # e.g. "blocks_operator_actions,hanoi,visit_all"
    if domain_name == "all":
        allowed_domains = None  # accept all
    elif "," in domain_name:
        allowed_domains = [d.strip() for d in domain_name.split(",")]
        print(f"Domain scope restricted to: {allowed_domains}")
    else:
        allowed_domains = [domain_name]
    
    domains = {}
    for problem_num, problem in enumerate(dataset["problems"]):
        problem_domain = problem["domain_name"]
        if allowed_domains is None or any(ad in problem_domain for ad in allowed_domains):
            if problem_domain not in domains:
                domains[problem_domain] = []
            for state_num in range(len(problem["states"])):
                domains[problem_domain].append((problem_num, state_num))
    
    print(f"Matched domains: {list(domains.keys())} (from domain_name='{domain_name}')")

    num_domains = len(domains)
    if num_domains == 0:
        print("No domains found matching the specified domain name.")
        return

    # Distribute tests among domains
    samples_per_domain = num_tests // num_domains
    remainder = num_tests % num_domains

    sampled_states = []
    domain_list = list(domains.items())
    for i, (domain, state_list) in enumerate(domain_list):
        num_samples = samples_per_domain + (1 if i < remainder else 0)
        num_samples = min(num_samples, len(state_list))
        sampled_states.extend(random.sample(state_list, num_samples))

    # Shuffle sampled states to randomize order
    random.shuffle(sampled_states)
    print("List of sampled states: ", sampled_states)
    print("Number of actual samples: ", len(sampled_states))
    print(f"Resume mode: {RESUME_RUN}")
    print(f"Oracle stage: {oracle_stage}")

    if dry_run:
        print("[DRY RUN] Enumerating selected work only; no API calls will be made.")
        for id_counter, (problem_id, state_num) in enumerate(sampled_states):
            problem = dataset["problems"][problem_id]
            request_id = f"{id_counter}_problem_{problem_id}_state_{state_num}_domain_{problem['domain_name']}"
            print(
                json.dumps(
                    {
                        "request_id": request_id,
                        "problem_id": problem_id,
                        "state_num": state_num,
                        "domain_name": problem["domain_name"],
                        "domain_file": problem["domain_file"],
                        "oracle_stage": oracle_stage,
                    }
                )
            )
        return

    id_counter = 0
    evaluation_output_file_objects = os.path.join(output_folder, output_filename_objects)
    evaluation_output_file_atoms = os.path.join(output_folder, output_filename_atoms)
    evaluation_output_file_goals = os.path.join(output_folder, output_filename_goals)

    if not api_keys:
        api_keys = [OPENAI_API_KEY]

    api_key_index = 0

    if not RESUME_RUN:
        for _eval_file in [evaluation_output_file_objects, evaluation_output_file_atoms, evaluation_output_file_goals]:
            open(_eval_file, 'w').close()
        for _full_resp_file in [FULL_RESPONSES_OBJECTS_FILE, FULL_RESPONSES_ATOMS_FILE, FULL_RESPONSES_GOALS_FILE]:
            open(os.path.join(output_folder, _full_resp_file), 'w').close()

    _evaluation_write_mode = 'a' if RESUME_RUN else 'w'
    with open(evaluation_output_file_objects, _evaluation_write_mode) as f_objects, \
         open(evaluation_output_file_atoms, _evaluation_write_mode) as f_atoms, \
         open(evaluation_output_file_goals, _evaluation_write_mode) as f_goals:

        for problem_id, state_num in sampled_states:
            # Resume: skip samples that already have output files
            _domain_name_for_resume = dataset["problems"][problem_id]["domain_name"]
            _resume_id = f"{id_counter}_problem_{problem_id}_state_{state_num}_domain_{_domain_name_for_resume}"
            _resume_obj = os.path.join(output_folder, f"{_resume_id}_objects.json")
            _resume_atoms = os.path.join(output_folder, f"{_resume_id}_atoms.json")
            _resume_goals = os.path.join(output_folder, f"{_resume_id}_goals.json")
            _state_for_resume = dataset["problems"][problem_id]["states"][state_num]
            _needs_goal_output = bool(_state_for_resume.get("goal_predicates"))
            _has_goals = os.path.exists(_resume_goals) and os.path.getsize(_resume_goals) > 0
            if RESUME_RUN and os.path.exists(_resume_obj) and os.path.getsize(_resume_obj) > 0 \
                and os.path.exists(_resume_atoms) and os.path.getsize(_resume_atoms) > 0 \
                and ((not _needs_goal_output) or _has_goals):
                print(f"[RESUME] Skipping {_resume_id} — output files already exist")
                id_counter += 1
                continue

            api_key = api_keys[api_key_index % len(api_keys)]
            api_key_index += 1

            # try:
            process_sample(
                dataset=dataset,
                dataset_path=dataset_path,
                problem_id=problem_id,
                state_num=state_num,
                num_examples=num_examples,
                id_counter=id_counter,
                evaluation_results_objects=evaluation_results_objects,
                evaluation_results_atoms=evaluation_results_atoms,
                evaluation_results_goals=evaluation_results_goals,
                api_key=api_key,
                num_answers=num_answers_per_sample,
                use_probs=use_probs,
                obs_threshold=obs_threshold,
                use_ground_truth=use_ground_truth,
                oracle_stage=oracle_stage
            )
            # except KeyboardInterrupt:
            #     print("Interrupted by user.")
            #     raise KeyboardInterrupt
            # except Exception as e:
            #     print("maybe token limit")
            #     print(e)
            #     time.sleep(5)
            #     continue
                
            id_counter += 1

            # Add model provenance to each result before writing
            _provenance = {
                "model_to_use": os.getenv("MODEL_TO_USE", "unknown"),
                "model_name": os.getenv("MODEL_NAME", "unknown"),
                "config_path": CONFIG_PATH,
                "oracle_stage": oracle_stage,
            }

            # Write object evaluation results
            for eval_result in evaluation_results_objects:
                eval_result.update(_provenance)
                json.dump(eval_result, f_objects)
                f_objects.write('\n')
            f_objects.flush()
            evaluation_results_objects.clear()

            # Write predicate evaluation results
            for eval_result in evaluation_results_atoms:
                eval_result.update(_provenance)
                json.dump(eval_result, f_atoms)
                f_atoms.write('\n')
            f_atoms.flush()
            evaluation_results_atoms.clear()

            # Write goals evaluation results (if any)
            for eval_result in evaluation_results_goals:
                eval_result.update(_provenance)
                json.dump(eval_result, f_goals)
                f_goals.write('\n')
            f_goals.flush()
            evaluation_results_goals.clear()
            # Rate-limit mitigation: small delay between samples
            time.sleep(3)

    # After all samples are processed, load all results and compute metrics
    evaluation_results_objects_all = []
    evaluation_results_atoms_all = []
    evaluation_results_goals_all = []

    # Ensure evaluation files exist (may be missing if process was interrupted)
    for _ef in [evaluation_output_file_objects, evaluation_output_file_atoms, evaluation_output_file_goals]:
        if not os.path.exists(_ef):
            open(_ef, 'w').close()

    with open(evaluation_output_file_objects, 'r') as f_objects, \
         open(evaluation_output_file_atoms, 'r') as f_atoms, \
         open(evaluation_output_file_goals, 'r') as f_goals:

        for line in f_objects:
            evaluation_results_objects_all.append(json.loads(line))
        for line in f_atoms:
            evaluation_results_atoms_all.append(json.loads(line))
        for line in f_goals:
            evaluation_results_goals_all.append(json.loads(line))

    # Compute and print final metrics
    metrics_objects = calculate_metrics_objects(evaluation_results_objects_all)
    print("Objects Evaluation Metrics:")
    print(json.dumps(metrics_objects, indent=2))

    metrics_atoms = calculate_metrics(evaluation_results_atoms_all)
    print("Atoms Evaluation Metrics:")
    print(json.dumps(metrics_atoms, indent=2))

    metrics_goals = calculate_metrics(evaluation_results_goals_all)
    print("Goals Evaluation Metrics:")
    print(json.dumps(metrics_goals, indent=2))


def main():
    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # API keys for OpenAI
    api_keys = [OPENAI_API_KEY]

    create_sequential_calls_from_dataset_for_domain(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        domain_name=DOMAIN_NAME,
        num_examples=NUM_EXAMPLES,
        num_tests=NUM_SAMPLES,
        output_folder=OUTPUT_FOLDER,
        api_keys=api_keys,
        num_answers_per_sample=NUM_ANSWERS_PER_SAMPLE,
        use_probs=False,
        obs_threshold=0.05,
        use_ground_truth=False,
        oracle_stage=ORACLE_STAGE,
        dry_run=_args.dry_run,
    )


if __name__ == "__main__":
    main()
