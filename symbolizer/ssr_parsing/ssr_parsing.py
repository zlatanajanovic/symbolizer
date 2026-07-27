import os
import json
import base64
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator
from collections import defaultdict
import threading
import random
from itertools import product 
from PIL import Image
import sys
import os
import numpy as np

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Assuming these utility functions are available from your existing codebase
from .parsing_utils.general_utils import load_env, load_config
from .parsing_utils.chatgpt_utils import (
    get_object_schema_from_pddl,
     encode_image_base64,
)
from .parsing_utils.generate_scenes_utils import get_atom_schema_from_chatgpt_objects

from pddl import parse_domain

# OpenAI Imports
from mistralai import Mistral

from .run_eval_vlm import load_and_create_example_from_dataset, get_predicates_structured, load_dataset, make_messages_image, get_structured_output_parsed, make_messages_text, get_predict_next_state_prompt, load_and_create_example_from_dataset_prediction, get_all_grounded_atoms_prompt

# Initialize OpenAI client (ensure environment variables are loaded)
config = load_config('./config.yaml')
load_env(config['env_file'])


MAX_TOKENS = config['max_tokens']
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_TO_USE = os.getenv("MODEL_TO_USE", "None Set")
SELF_HOSTED_ENDPOINT = os.getenv("SELF_HOSTED_ENDPOINT", "None Set")
USE_LOGPROBS = os.getenv("USE_LOGPROBS", "false").lower() == "true"
DEFAULT_TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5" if USE_LOGPROBS else "0"))

# Rate-Limiting (Adjust according to TODO check if neededrate limits)
api_key_last_called = defaultdict(lambda: 0.0)
api_key_lock = threading.Lock()


class ActionRanking(BaseModel):
    action: str
    score: float

    @field_validator("score")
    @classmethod
    def _score_within_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("score must be between 0 and 1")
        return value


class BestActionResponse(BaseModel):
    best_action: str
    action_rankings: List[ActionRanking] = Field(default_factory=list)


class HeuristicScoreResponse(BaseModel):
    heuristic_score: int
    explanation: str = Field(default="", description="Explanation of the heuristic score")

    @field_validator("heuristic_score")
    @classmethod
    def _score_within_bounds(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("heuristic_score must be between 1 and 10")
        return value



def get_simplified_object_probs(
    objects_probs,
    object_threshold_propability: float = 0.05,
    ):
    if not objects_probs:
        return []

    # When logprobs are available, filter by exp(logprob) as before.
    if USE_LOGPROBS:
        objects_probs_new = [
            v for v in objects_probs
            if "logprob" in v and np.exp(v["logprob"]) > object_threshold_propability
        ]
        set_names = set()
        objects_probs_new_unique = []
        sum_all_probs = 0

        for object_dict in objects_probs_new:
            if object_dict["object"].name not in set_names:
                set_names.add(object_dict["object"].name)
                objects_probs_new_unique.append(object_dict)
                sum_all_probs += np.exp(object_dict["logprob"])

        object_propabilities = []

        for object_dict in objects_probs_new_unique:
            object_instance = object_dict["object"]
            object_softmax_prob = np.exp(object_dict["logprob"])  # / sum_all_probs
            object_propabilities.append(
                {
                    "object": object_instance,
                    "prob": float(object_softmax_prob)
                }
            )
        return object_propabilities

    # Fallback path for providers/models that do not expose token logprobs:
    # keep all returned objects with unit confidence so downstream planning can proceed.
    object_propabilities = []
    seen = set()
    for object_dict in objects_probs:
        obj = object_dict.get("object")
        if obj is None:
            continue
        key = str(obj)
        if key in seen:
            continue
        seen.add(key)
        p = object_dict.get("prob", 1.0)
        object_propabilities.append({"object": obj, "prob": float(p)})
    return object_propabilities
        

##############################
# Only addition: we add an optional parameter n_answers=1
##############################
def get_simplified_pred_probs(
    pred_probs,
    pred_threshold_propability: float = 0.05,
    n_answers: int = 1
    ):
    """
    For Mistral, if n_answers == 1 => all probabilities are 0.8.
    If n_answers > 1 => we treat the fraction (#found / n_answers)*0.8 as the probability.
    For others, the original logic is used.
    """
    if USE_LOGPROBS and MODEL_TO_USE != "mistral":
        pred_probs_new = [v for v in pred_probs if np.exp(v["logprob"]) > pred_threshold_propability] 
        set_names = set()
        preds_probs_new_unique = []
        sum_all_probs = 0

        for pred_dict in pred_probs_new:
            if pred_dict["predicate"].__str__() not in set_names:
                set_names.add(pred_dict["predicate"].__str__())
                preds_probs_new_unique.append(pred_dict)
                sum_all_probs += np.exp(pred_dict["logprob"])

        pred_propabilities = []

        for pred_dict in preds_probs_new_unique:
            pred_instance = pred_dict["predicate"]
            pred_softmax_prob = np.exp(pred_dict["logprob"]) # / sum_all_probs (optional normalization)
            pred_propabilities.append(
                {
                    "predicate": pred_instance,
                    "prob": float(pred_softmax_prob)
                }
            )
    else:
        # Fallback path for providers/models that do not expose token logprobs:
        # keep all returned predicates with unit confidence so downstream planning can proceed.
        pred_propabilities = []
        seen = set()
        for pred_dict in pred_probs:
            pred = pred_dict.get("predicate")
            if pred is None:
                continue
            key = str(pred)
            if key in seen:
                continue
            seen.add(key)
            p = pred_dict.get("prob", 1.0)
            pred_propabilities.append({"predicate": pred, "prob": float(p)})


## Logprobs stuff PRELIMINARY
def get_logprobs_list_from_completion(completion,
                                      iteration_number:int = 0):
    """
    Extract log probabilities from the completion.
    """
    list_all_logprobs = []
    for logprob in completion.choices[iteration_number].logprobs.content:
        token = logprob.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
        log_probability = logprob.logprob
        top_probs_dict = {
            top_logprob.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", ""): top_logprob.logprob
            for top_logprob in logprob.top_logprobs
        }
        list_all_logprobs.append(
            {"token": token, "log_probability": log_probability, "top_probs": top_probs_dict}
        )
    return list_all_logprobs

def get_object_probabilities_separately(response: Dict[str, Any], object_schema, iteration_number:int = 0):
    """
    Given a structured completion response (with logprobs) and an object schema,
    parse the logprobs to extract all possible names and types along with their probabilities.
    """
    if not USE_LOGPROBS:
        return {"names": [], "types": []}
        

    if not response.choices:
        print("No choices in response")
        return {"names": [], "types": []}

    logprob_list = getattr(response.choices[iteration_number], "logprobs", None)
    if not logprob_list or logprob_list.content is None:
        print("No logprobs in response")
        return {"names": [], "types": []}
    logprob_list = logprob_list.content

    name_options = []
    type_options = []

    i = 0
    while i < len(logprob_list):
        token_info = logprob_list[i]
        token_str = token_info.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")

        if i >= 2:
            token_str_full = logprob_list[i-2].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") + \
                             logprob_list[i-1].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") + \
                             logprob_list[i].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
            

            # Check for 'name":"'
            if token_str_full.endswith('name":"'):
                obj_tokens = []
                obj_logprob_sum = 0.0
                j = i + 1
                while j < len(logprob_list):
                    current_token = logprob_list[j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                    if '"' in current_token:
                        break
                    obj_tokens.append(logprob_list[j])
                    obj_logprob_sum += logprob_list[j].logprob
                    j += 1
                object_name = "".join([t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens]).strip()
                name_options.append({
                    "name": object_name,
                    "logprob": obj_logprob_sum,
                    "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens]
                })
                i = j
                continue

            # Check for 'type":"'
            elif token_str_full.endswith('type":"'):
                obj_tokens = []
                obj_logprob_sum = 0.0
                k = i + 1
                while k < len(logprob_list):
                    current_token = logprob_list[k].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                    if '"' in current_token:
                        break
                    obj_tokens.append(logprob_list[k])
                    obj_logprob_sum += logprob_list[k].logprob
                    k += 1
                object_type = "".join([t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens]).strip()
                type_options.append({
                    "type": object_type,
                    "logprob": obj_logprob_sum,
                    "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens],
                    "object_name": "UNUSED"
                })
                i = k
                continue

        i += 1

    return {
        "names": name_options,
        "types": type_options
    }

def get_object_probabilities(response: Dict[str, Any], object_schema,iteration_number:int = 0):
    """
    Generate all valid object combinations (name and type) from the response.
    """
    if not USE_LOGPROBS:
        return []
    
    extracted = get_object_probabilities_separately(response, object_schema,iteration_number=iteration_number)
    name_options = extracted.get("names", [])
    type_options = extracted.get("types", [])
    
    
    if not name_options or not type_options:
        return []
    
    combined_objects = []
    for name_opt, type_opt in product(name_options, type_options):
        name_str = name_opt['name']
        type_str = type_opt['type']
        combined_logprob = name_opt['logprob'] + type_opt['logprob']
        
        obj_dict = {
            "name": name_str,
            "type": type_str
        }
        
        try:
            validated = object_schema(**{"objects":[obj_dict]})
            validated_object = validated.objects[0]
            combined_objects.append({
                "object": validated_object,
                "logprob": combined_logprob,
                "tokens": {
                    "name": name_opt['tokens'],
                    "type": type_opt['tokens'],
                }
            })
        except ValidationError as e:
            print(f"Validation error: {e}")
            continue
    
    combined_objects.sort(key=lambda x: x['logprob'], reverse=True)
    
    return combined_objects

def get_predicate_probabilities_separately(response: Dict[str, Any], predicate_schema,iteration_number:int = 0):
    """
    Given a structured completion response (with logprobs) and a predicate schema,
    parse the logprobs to extract possible fields. (Simplified example.)
    """
    if not USE_LOGPROBS:
        return {"predicate_type": [], "x": [], "y": [], "z": []}
    if not response.choices:
        print("No choices in response")
        return {"predicate_type": [], "x": [], "y": [], "z": []}

    logprobs_attr = getattr(response.choices[iteration_number], "logprobs", None)
    if not logprobs_attr or logprobs_attr.content is None:
        print("No logprobs in response")
        return {"predicate_type": [], "x": [], "y": [], "z": []}
    logprob_list = logprobs_attr.content

    predicate_type_options = []
    x_options = []
    y_options = []
    z_options = []

    i = 0
    predicate_type_str = ""
    x_name_str = ""
    y_name_str = ""
    z_name_str = ""

    while i < len(logprob_list):
        token_info = logprob_list[i]
        token_str = token_info.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
        ## ONLY FIX FOR BPE BASED GPT2 LIKE QWEN MODEL
        token_str = token_str.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
        look_behind_length = min(i + 1, 4)
        token_str_full = "".join(logprob_list[i - look_behind_length + 1 : i + 1][j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                                 for j in range(look_behind_length)).replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")

        if '"predicate_type":"' in token_str_full:
            obj_tokens = []
            obj_logprob_sum = 0.0
            j = i + 1
            while j < len(logprob_list):
                current_token = logprob_list[j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                if '"' in current_token:
                    break
                obj_tokens.append(logprob_list[j])
                obj_logprob_sum += logprob_list[j].logprob
                j += 1
            predicate_type_str = "".join(t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens).strip()
            predicate_type_options.append({
                "predicate_type": predicate_type_str,
                "logprob": obj_logprob_sum,
                "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens]
            })
            i = j
            continue

        if 'x"' in token_str_full and token_str_full.endswith('name":"'):
            obj_tokens = []
            obj_logprob_sum = 0.0
            j = i + 1
            while j < len(logprob_list):
                current_token = logprob_list[j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                if '"' in current_token:
                    break
                obj_tokens.append(logprob_list[j])
                obj_logprob_sum += logprob_list[j].logprob
                j += 1
            x_name_str = "".join(t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens).strip()
            x_options.append({
                "name": x_name_str,
                "logprob": obj_logprob_sum,
                "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens],
                "predicate_name": predicate_type_str
            })
            i = j
            continue

        if 'y"' in token_str_full and token_str_full.endswith('name":"'):
            obj_tokens = []
            obj_logprob_sum = 0.0
            j = i + 1
            while j < len(logprob_list):
                current_token = logprob_list[j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                if '"' in current_token:
                    break
                obj_tokens.append(logprob_list[j])
                obj_logprob_sum += logprob_list[j].logprob
                j += 1
            y_name_str = "".join(t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens).strip()
            y_options.append({
                "name": y_name_str,
                "logprob": obj_logprob_sum,
                "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens],
                "predicate_name": predicate_type_str,
                "x_name": x_name_str
            })
            i = j
            continue

        if 'z"' in token_str_full and token_str_full.endswith('name":"'):
            obj_tokens = []
            obj_logprob_sum = 0.0
            j = i + 1
            while j < len(logprob_list):
                current_token = logprob_list[j].token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "")
                if '"' in current_token:
                    break
                obj_tokens.append(logprob_list[j])
                obj_logprob_sum += logprob_list[j].logprob
                j += 1
            z_name_str = "".join(t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens).strip()
            z_options.append({
                "name": z_name_str,
                "logprob": obj_logprob_sum,
                "tokens": [t.token.replace("Ġ", "").replace("Ċ", "").replace("\u0120", "").replace("\u010a", "") for t in obj_tokens],
                "predicate_name": predicate_type_str,
                "x_name": x_name_str,
                "y_name": y_name_str
            })
            i = j
            continue

        i += 1

    return {
        "predicate_type": predicate_type_options,
        "x": x_options,
        "y": y_options,
        "z": z_options
}


def get_predicate_probabilities(response: Dict[str, Any], predicate_schema,iteration_number:int = 0):
    """
    Generate all valid predicates (predicate_type, x, y, z) from the response.
    """
    if not USE_LOGPROBS:
        return []
    extracted = get_predicate_probabilities_separately(response, predicate_schema,iteration_number=iteration_number)
    pred_type_options = extracted["predicate_type"]
    x_options = extracted["x"]
    x_options.append({"name": "Empty", "logprob": 0.0, "tokens": []})
    y_options = extracted["y"]
    y_options.append({"name": "Empty", "logprob": 0.0, "tokens": []})
    z_options = extracted["z"]
    z_options.append({"name": "Empty", "logprob": 0.0, "tokens": []})

    if not pred_type_options:
        return []

    combined_predicates = []
    for pred_type_opt in pred_type_options:
        for x_opt in (x_options if x_options else [{"name": None, "logprob": 0.0, "tokens": []}]):
            for y_opt in (y_options if y_options else [{"name": None, "logprob": 0.0, "tokens": []}]):
                for z_opt in (z_options if z_options else [{"name": None, "logprob": 0.0, "tokens": []}]):
                    combined_logprob = pred_type_opt["logprob"] + \
                                       x_opt["logprob"] + \
                                       y_opt["logprob"] + \
                                       z_opt["logprob"]

                    candidate_dict = {"predicate_type": pred_type_opt["predicate_type"]}
                    if x_opt["name"] is not None and x_opt["name"] != "Empty":
                        if x_opt["predicate_name"] == pred_type_opt["predicate_type"]:
                            candidate_dict["x"] = {"name": x_opt["name"]}
                        else:
                            continue
                        if y_opt["name"] is not None and y_opt["name"] != "Empty":
                            if y_opt["predicate_name"] == pred_type_opt["predicate_type"] and y_opt["x_name"] == x_opt["name"]:
                                candidate_dict["y"] = {"name": y_opt["name"]}
                            else:
                                continue
                            if z_opt["name"] is not None and z_opt["name"] != "Empty":
                                if (z_opt["predicate_name"] == pred_type_opt["predicate_type"] 
                                    and z_opt["x_name"] == x_opt["name"] 
                                    and z_opt["y_name"] == y_opt["name"]):
                                    candidate_dict["z"] = {"name": z_opt["name"]}
                                else:
                                    continue

                    try:
                        validated = predicate_schema(**{"grounded_predicates": [candidate_dict]})
                        validated_predicate = validated.grounded_predicates[0]
                        combined_predicates.append({
                            "predicate": validated_predicate,
                            "logprob": combined_logprob,
                            "tokens": {
                                "predicate_type": pred_type_opt["tokens"],
                                "x": x_opt["tokens"],
                                "y": y_opt["tokens"],
                                "z": z_opt["tokens"],
                            }
                        })
                    except ValidationError as e:
                        continue

    combined_predicates.sort(key=lambda x: x["logprob"], reverse=True)
    return combined_predicates


def retrieve_objects(
    image_path: str,
    domain_name: str,
    dataset: Dict[str, Any],
    num_examples: int = 1,
    dataset_path: str = "path/to/master_dataset.json",
    number_answers: int = 1,
    problem_id = -1,
    problem_file: str = None,
    instruction_text: str = None,
    _out_usage: Optional[dict] = None,
) -> Any:
    """
    Retrieves objects from an image based on the domain using few-shot examples.
    """
    example_history_objects, _, _ = load_and_create_example_from_dataset(
        dataset_path, 
        problem_num=problem_id, 
        num_examples=num_examples
    )
    image_base64_curr = encode_image_base64(image_path=image_path)
    domain_file = next(
        problem["domain_file"] for problem in dataset["problems"] 
        if problem["domain_name"] == domain_name
    )

    # Keep grounding constrained only by object types (not fixed object names).
    object_schema_curr, list_types_curr = get_object_schema_from_pddl(domain_file)
    prompt_get_objects_curr = (
        f"Analyze the image and retrieve all objects that match the following types: {list_types_curr}. "
    )

    if instruction_text:
        prompt_get_objects_curr += (
            "\nTask instruction (STRICT: include ONLY task-relevant objects):\n"
            f"{instruction_text.strip()}"
        )
    
    message_get_object_curr = make_messages_image(prompt_get_objects_curr, image_base64_curr)
    messages_get_object = [*example_history_objects, *message_get_object_curr]
    
    messages_get_object_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_get_object
    ]
    response_format_objects = object_schema_curr
    system_message = {
      "role": "system",
      "content": "Your goal is to retrieve objects. It is very important to keep consistent..."
    }

    response_objects, object_list_instance = get_structured_output_parsed(
        messages=messages_get_object_together,
        schema_response=response_format_objects,
        top_logprobs = DEFAULT_TOP_LOGPROBS, 
        number_answers = number_answers
    )
    if _out_usage is not None and response_objects is not None:
        try:
            raw_dict = getattr(response_objects, "to_dict", lambda: {})()
            gem_usage = raw_dict.get("_gemini_usage", {}) if isinstance(raw_dict, dict) else {}
            gem_ctx = raw_dict.get("_gemini_context_cache", {}) if isinstance(raw_dict, dict) else {}
            _out_usage["prompt_tokens"] = int(gem_usage.get("prompt_tokens", 0) or 0)
            _out_usage["completion_tokens"] = int(gem_usage.get("completion_tokens", 0) or 0)
            _out_usage["total_tokens"] = int(gem_usage.get("total_tokens", 0) or 0)
            _out_usage["cached_content_tokens"] = int(gem_usage.get("cached_content_tokens", 0) or 0)
            _out_usage["context_cache_enabled"] = bool(gem_ctx.get("enabled", False))
            _out_usage["context_cache_variant"] = int(gem_ctx.get("variant", 0) or 0)
            _out_usage["context_cache_name"] = str(gem_ctx.get("cached_content_name", "") or "")
            _out_usage["available"] = bool(gem_usage.get("available", False))
        except Exception:
            pass

    if object_list_instance is None:
        raise RuntimeError(
            f"Failed to retrieve objects (provider={MODEL_TO_USE}, model={MODEL_NAME}, USE_LOGPROBS={USE_LOGPROBS})."
        )

    if MODEL_TO_USE == "mistral":
        raise RuntimeError(
            "Mistral does not support logprobs — cannot compute real object probabilities. "
            "Use a logprobs-capable model (e.g., Gemini)."
        )

    if not USE_LOGPROBS:
        # No-logprobs fallback: return unit probabilities for parsed objects.
        objects_raw = getattr(object_list_instance, "objects", []) or []
        object_probs = [{"object": obj, "prob": 1.0} for obj in objects_raw]
        return [object_list_instance], [object_probs]

    # ------------------- ADDED BLOCK FOR SELF-HOSTED (same as openai) -------------------
    if MODEL_TO_USE == "selfhosted":
        if number_answers == 1:
            object_probs = get_object_probabilities(response_objects, response_format_objects)
            
            return [object_list_instance],[object_probs]
        else:
            object_list_instance_mul = []
            object_probs_mul = []
            for i in range(number_answers):
                object_probs_i = get_object_probabilities(response_objects, response_format_objects, iteration_number=i)
                object_probs_mul.append(object_probs_i)
                object_list_instance_i = response_objects.choices[i].message.parsed
                object_list_instance_mul.append(object_list_instance_i)
            return object_list_instance_mul, object_probs_mul
    # ------------------------------------------------------------------------------------

    # The original openai-based code path:
    if number_answers == 1:
        object_probs = get_object_probabilities(response_objects,response_format_objects)
        return [object_list_instance],[object_probs]
    else:
        object_list_instance_mul = []
        object_probs_mul = []
        for i in range(number_answers):
            object_probs_i = get_object_probabilities(response_objects,response_format_objects,iteration_number=i)
            object_probs_mul.append(object_probs_i)
            object_list_instance_i = response_objects.choices[i].message.parsed
            object_list_instance_mul.append(object_list_instance_i)
        return object_list_instance_mul,object_probs_mul


def retrieve_predicates(
    image_path: str,
    domain_name: str,
    objects: Any,
    dataset: Dict[str, Any],
    num_examples: int = 1,
    dataset_path: str = "path/to/master_dataset.json",
    number_answers: int = 1,
    problem_id = -1,
    instruction_text: str = None,
    _out_logprobs: Optional[dict] = None,
    _out_usage: Optional[dict] = None,
) -> Any:
    """
    Retrieves predicates from an image based on the domain and identified objects using few-shot examples.
    """
    _, example_history_atoms, _ = load_and_create_example_from_dataset(
        dataset_path, 
        problem_num=problem_id, 
        num_examples=num_examples
    )
    
    image_base64_curr = encode_image_base64(image_path=image_path)
    domain_file = next(
        problem["domain_file"] for problem in dataset["problems"] 
        if problem["domain_name"] == domain_name
    )

    predicates_structured = get_predicates_structured(objects, domain_file)
    predicates_structured = {k: predicates_structured[k] for k in sorted(predicates_structured)}
    prompt_get_atoms_curr = get_all_grounded_atoms_prompt(
        predicates_structured=predicates_structured,
        object_list=objects,
        schema_json=""
    )
    if instruction_text:
        prompt_get_atoms_curr += (
            "\nTask instruction (STRICT: include ONLY task-relevant grounded predicates/atoms):\n"
            f"{instruction_text.strip()}"
        )
    system_message = {
      "role": "system",
      "content": (
          "You are a symbolic state extractor for planning. "
          "Output only schema-valid JSON, reusing object symbols exactly as provided, "
          "and return an exhaustive, non-contradictory set of true grounded predicates."
      )
    }

    message_get_atoms_curr = make_messages_image(prompt_get_atoms_curr, image_base64_curr)
    messages_get_atoms = [system_message,*example_history_atoms, *message_get_atoms_curr]
    
    messages_get_atoms_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_get_atoms
    ]
    Atoms_schema_chatGPT = get_atom_schema_from_chatgpt_objects(objects, domain_file)
    response_format_atoms = Atoms_schema_chatGPT

    response_atoms, grounded_atoms_instance = get_structured_output_parsed(
        messages=messages_get_atoms_together,
        schema_response=response_format_atoms,
        top_logprobs = DEFAULT_TOP_LOGPROBS,
        number_answers = number_answers
    )
    if _out_usage is not None and response_atoms is not None:
        try:
            raw_dict = getattr(response_atoms, "to_dict", lambda: {})()
            gem_usage = raw_dict.get("_gemini_usage", {}) if isinstance(raw_dict, dict) else {}
            gem_ctx = raw_dict.get("_gemini_context_cache", {}) if isinstance(raw_dict, dict) else {}
            _out_usage["prompt_tokens"] = int(gem_usage.get("prompt_tokens", 0) or 0)
            _out_usage["completion_tokens"] = int(gem_usage.get("completion_tokens", 0) or 0)
            _out_usage["total_tokens"] = int(gem_usage.get("total_tokens", 0) or 0)
            _out_usage["cached_content_tokens"] = int(gem_usage.get("cached_content_tokens", 0) or 0)
            _out_usage["context_cache_enabled"] = bool(gem_ctx.get("enabled", False))
            _out_usage["context_cache_variant"] = int(gem_ctx.get("variant", 0) or 0)
            _out_usage["context_cache_name"] = str(gem_ctx.get("cached_content_name", "") or "")
            _out_usage["available"] = bool(gem_usage.get("available", False))
        except Exception:
            pass

    # Expose raw logprobs to caller via mutable sink dict
    if _out_logprobs is not None and response_atoms is not None:
        try:
            choice0 = response_atoms.choices[0]
            lp = getattr(choice0, "logprobs", None)
            if lp is not None and hasattr(lp, "content") and lp.content:
                _out_logprobs["avg_logprobs"] = getattr(
                    response_atoms, "_avg_logprobs", None
                )
                _out_logprobs["tokens"] = [
                    {"token": t.token, "log_probability": t.logprob}
                    for t in lp.content
                ]
            # For Gemini, avg_logprobs may be stored in the raw dict
            raw_dict = getattr(response_atoms, "to_dict", lambda: {})()
            gemini_lp = raw_dict.get("_gemini_logprobs", {}) if isinstance(raw_dict, dict) else {}
            if gemini_lp.get("available"):
                _out_logprobs["avg_logprobs"] = gemini_lp.get("avg_logprobs")
                if not _out_logprobs.get("tokens") and gemini_lp.get("tokens"):
                    _out_logprobs["tokens"] = gemini_lp["tokens"]
        except Exception:
            pass  # logprobs extraction is best-effort

    if grounded_atoms_instance is None:
        raise RuntimeError(
            f"Failed to retrieve predicates (provider={MODEL_TO_USE}, model={MODEL_NAME}, USE_LOGPROBS={USE_LOGPROBS})."
        )

    if MODEL_TO_USE == "mistral":
        raise RuntimeError(
            "Mistral does not support logprobs — cannot compute real predicate probabilities. "
            "Use a logprobs-capable model (e.g., Gemini)."
        )

    if not USE_LOGPROBS:
        # No-logprobs fallback: return unit probabilities for parsed predicates.
        preds_raw = getattr(grounded_atoms_instance, "grounded_predicates", []) or []
        predicate_probs = [{"predicate": pred, "prob": 1.0} for pred in preds_raw]
        return [grounded_atoms_instance], [predicate_probs]

    # ------------------- ADDED BLOCK FOR SELF-HOSTED (same as openai) -------------------
    if MODEL_TO_USE == "selfhosted":
        if number_answers == 1:
            predicate_probs = get_predicate_probabilities(response_atoms, response_format_atoms)
            return [grounded_atoms_instance],[predicate_probs]
        else:
            predicate_probs_mul = []
            predicate_list_instance_mul = []
            for i in range(number_answers):
                predicate_probs_i = get_predicate_probabilities(response_atoms, response_format_atoms,iteration_number=i)
                predicate_probs_mul.append(predicate_probs_i)
                predicate_list_instance_i = response_atoms.choices[i].message.parsed
                predicate_list_instance_mul.append(predicate_list_instance_i)
            return predicate_list_instance_mul,predicate_probs_mul
    # ------------------------------------------------------------------------------------

    if number_answers == 1:
        predicate_probs = get_predicate_probabilities(response_atoms, response_format_atoms)
        return [grounded_atoms_instance],[predicate_probs]
    else:
        predicate_probs_mul = []
        predicate_list_instance_mul = []
        for i in range(number_answers):
            predicate_probs_i = get_predicate_probabilities(response_atoms, response_format_atoms,iteration_number=i)
            predicate_probs_mul.append(predicate_probs_i)
            predicate_list_instance_i = response_atoms.choices[i].message.parsed
            predicate_list_instance_mul.append(predicate_list_instance_i)
        return predicate_list_instance_mul,predicate_probs_mul


def predict_next_state(
    action: str,
    current_state: Dict[str, Any],
    objects: Any,   
    domain_name: str,
    dataset: Dict[str, Any],
    num_examples: int = 1,
    dataset_path: str = "path/to/next_state_dataset.json",
    number_answers: int = 1,
    problem_id = -1
) -> Dict[str, Any]:
    """
    Predicts the next state given an action and the current symbolic state using few-shot examples.
    """
    if problem_id == -1:
        print(f"Problem ID not provided. Finding it from the dataset for domain name: {domain_name}")
        problem_id = find_problem_num(domain_name,load_dataset(dataset_path),dataset_path) #TODO: Check if correct
        print(f"Problem ID found: {problem_id}")
    example_history = load_and_create_example_from_dataset_prediction(
        dataset_path, 
        problem_num=problem_id, 
        num_examples=num_examples
    )
    print(f"Example history loaded:")
    domain_problems = [
        idx for idx, prob in enumerate(dataset["problems"]) 
        if prob["problem_name"] == domain_name
    ]
    if not domain_problems:
        print(f"No problems found for domain: {domain_name}")
        return None
    
    selected_problem_num = random.choice(domain_problems)
    domain_file = dataset["problems"][selected_problem_num]["domain_file"]
    atoms_curr = current_state.get("grounded_predicates", [])
    atoms_curr_str = json.dumps(atoms_curr)
    Atoms_schema_chatGPT = get_atom_schema_from_chatgpt_objects(objects, domain_file)
    print("Getting predicates structured...")
    predicates_structured = get_predicates_structured(objects, domain_file)
    print("got predicates structured...")
    prompt_predict_next_state = get_predict_next_state_prompt(
        list_types=[],  
        object_schema_json="",  
        predicates_structured=predicates_structured,
        object_list=objects,
        atoms_schema_json="",
        action=action,
        atoms_curr=current_state
    )
    message_predict_next_state = make_messages_text(prompt_predict_next_state)
    messages_predict_next_state = [*example_history, *message_predict_next_state]
    response_format = Atoms_schema_chatGPT
    messages_predict_next_state_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_predict_next_state
    ]

    response_next_state, predicted_atoms_instance = get_structured_output_parsed(
        messages=messages_predict_next_state_together,
        schema_response=response_format,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
        number_answers=number_answers
    )

    if predicted_atoms_instance is None:
        print("Failed to predict the next state.")
        return None

    if MODEL_TO_USE == "mistral":
        raise RuntimeError(
            "Mistral does not support logprobs — cannot compute real predicate probabilities. "
            "Use a logprobs-capable model (e.g., Gemini)."
        )

    if not USE_LOGPROBS:
        # No-logprobs fallback for next-state prediction.
        preds_raw = getattr(predicted_atoms_instance, "grounded_predicates", []) or []
        predicted_atoms_probs = [{"predicate": pred, "prob": 1.0} for pred in preds_raw]
        return [predicted_atoms_instance], [predicted_atoms_probs]

    # ------------------- ADDED BLOCK FOR SELF-HOSTED (same as openai) -------------------
    if MODEL_TO_USE == "selfhosted":
        if number_answers == 1:
            predicted_atoms_probs = get_predicate_probabilities(response_next_state, response_format)
            return [predicted_atoms_instance], [predicted_atoms_probs]
        else:
            predicted_atoms_instance_mul = []
            predicted_atoms_probs_mul = []
            for i in range(number_answers):
                predicted_atoms_probs_i = get_predicate_probabilities(response_next_state, response_format, iteration_number=i)
                predicted_atoms_instance_i = response_next_state.choices[i].message.parsed
                predicted_atoms_instance_mul.append(predicted_atoms_instance_i)
                predicted_atoms_probs_mul.append(predicted_atoms_probs_i)
            return predicted_atoms_instance_mul, predicted_atoms_probs_mul
    # ------------------------------------------------------------------------------------
    if number_answers == 1:
        predicted_atoms_probs = get_predicate_probabilities(response_next_state, response_format)
        return [predicted_atoms_instance], [predicted_atoms_probs]
    else:
        predicted_atoms_instance_mul = []
        predicted_atoms_probs_mul = []
        for i in range(number_answers):
            predicted_atoms_probs_i = get_predicate_probabilities(response_next_state, response_format, iteration_number=i)
            predicted_atoms_instance_i = response_next_state.choices[i].message.parsed
            predicted_atoms_instance_mul.append(predicted_atoms_instance_i)
            predicted_atoms_probs_mul.append(predicted_atoms_probs_i)
        return predicted_atoms_instance_mul, predicted_atoms_probs_mul


def _goal_to_text(goal: Any) -> str:
    if goal is None:
        return "Unknown goal"
    if hasattr(goal, "literals"):
        return ", ".join(str(lit) for lit in goal.literals)
    if isinstance(goal, (list, tuple, set)):
        return ", ".join(str(item) for item in goal)
    return str(goal)


def _state_to_text(current_state: Dict[str, Any]) -> str:
    atoms = current_state.get("grounded_predicates", [])
    if not atoms:
        return "None"
    return "\n".join(f"- {atom}" for atom in atoms)


def select_best_action(
    actions: List[str],
    current_state: Dict[str, Any],
    goal: Any,
    domain_name: str,
    number_answers: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to pick the best action from the provided list and optionally rank them.
    """
    if not actions:
        return None

    goal_text = _goal_to_text(goal)
    state_text = _state_to_text(current_state)
    actions_text = "\n".join(f"- {action}" for action in actions)

    prompt = (
        f"You are acting in the domain '{domain_name}'.\n"
        f"Goal (desired predicates): {goal_text}\n\n"
        "Current grounded predicates:\n"
        f"{state_text}\n\n"
        "Available actions:\n"
        f"{actions_text}\n\n"
        "Select the single best action that most improves progress toward the goal. "
        "Rank all actions on a 0-1 scale (1 is best) and ensure the chosen best action "
        "appears in the rankings. Return JSON with keys 'best_action' and 'action_rankings'."
    )

    messages = make_messages_text(prompt)
    _, parsed = get_structured_output_parsed(
        messages=messages,
        schema_response=BestActionResponse,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
        number_answers=number_answers
    )

    if parsed is None:
        return None

    if hasattr(parsed, "model_dump"):
        parsed_dict = parsed.model_dump()
    else:
        parsed_dict = parsed

    best_action = parsed_dict.get("best_action")
    if best_action not in actions:
        # Safeguard: fall back to highest ranked valid action.
        for item in parsed_dict.get("action_rankings", []):
            candidate = item.get("action")
            if candidate in actions:
                best_action = candidate
                break
        else:
            best_action = actions[0]
    parsed_dict["best_action"] = best_action
    return parsed_dict


def score_state_heuristic(
    current_state: Dict[str, Any],
    goal: Any,
    domain_name: str,
    number_answers: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to assign a heuristic score from 1 (far) to 10 (goal achieved) for the current state.
    """
    goal_text = _goal_to_text(goal)
    state_text = _state_to_text(current_state)

    prompt = (
        f"You are evaluating progress in the domain '{domain_name}'.\n"
        f"Goal predicates: {goal_text}\n\n"
        "Current grounded predicates:\n"
        f"{state_text}\n\n"
        "Rate how close this state is to the goal on a 1-10 scale (1 = very far, 10 = goal achieved). "
        "Return JSON with 'heuristic_score' (integer 1-10) and 'explanation' describing the rating."
    )

    messages = make_messages_text(prompt)
    _, parsed = get_structured_output_parsed(
        messages=messages,
        schema_response=HeuristicScoreResponse,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
        number_answers=number_answers
    )

    if parsed is None:
        return None

    if hasattr(parsed, "model_dump"):
        parsed_dict = parsed.model_dump()
    else:
        parsed_dict = parsed

    score = parsed_dict.get("heuristic_score")
    if isinstance(score, (int, float)):
        parsed_dict["heuristic_score"] = int(max(1, min(10, round(score))))
    else:
        parsed_dict["heuristic_score"] = 1
    return parsed_dict


def retrieve_goal(
    instruction_text: str,
    domain_name: str,
    objects: Any,
    dataset: Dict[str, Any],
    num_examples: int = 1,
    dataset_path: str = "path/to/master_dataset.json",
    number_answers: int = 1,
    problem_id = -1
) -> Any:
    """
    Retrieves goal predicates from a textual instruction.
    """
    _, _, example_history_goals = load_and_create_example_from_dataset(
        dataset_path, 
        problem_num=problem_id, 
        num_examples=num_examples
    )

    domain_file = next(
        problem["domain_file"] for problem in dataset["problems"] 
        if problem["domain_name"] == domain_name
    )
    Atoms_schema_chatGPT = get_atom_schema_from_chatgpt_objects(objects, domain_file, include_negation=True)

    prompt_get_goals = (
        "Given the following instructions:\n"
        f"{instruction_text}\n"
        "Identify the goal states as a set of grounded predicates..."
    )
    
    message_get_goals_curr = make_messages_text(prompt_get_goals)
    messages_get_goals = [*example_history_goals, *message_get_goals_curr]
    
    messages_get_goals_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_get_goals
    ]

    response_format_goals = Atoms_schema_chatGPT
    
    response_goals, goals_instance = get_structured_output_parsed(
        messages=messages_get_goals_together,
        schema_response=response_format_goals,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
        number_answers=number_answers
    )

    if goals_instance is None:
        print("Failed to retrieve goals.")
        return [], []  # Return empty lists instead of None

    # ------------------- Mistral does not support logprobs -------------------
    if MODEL_TO_USE == "mistral":
        raise RuntimeError(
            "Mistral does not support logprobs — cannot compute real goal probabilities. "
            "Use a logprobs-capable model (e.g., Gemini)."
        )

    if not USE_LOGPROBS:
        # No-logprobs fallback: return unit probabilities for parsed goals.
        goals_raw = getattr(goals_instance, "grounded_predicates", []) or []
        goals_probs = [{"predicate": goal, "prob": 1.0} for goal in goals_raw]
        return [goals_instance], [goals_probs]

    # ------------------- ADDED BLOCK FOR SELF-HOSTED (same as openai) -------------------
    if MODEL_TO_USE == "selfhosted":
        if number_answers == 1:
            goals_probs = get_predicate_probabilities(response_goals, response_format_goals)
            return [goals_instance], [goals_probs]
        else:
            goals_instance_mul = []
            goals_probs_mul = []
            for i in range(number_answers):
                goals_probs_i = get_predicate_probabilities(response_goals, response_format_goals, iteration_number=i)
                goals_instance_i = response_goals.choices[i].message.parsed
                goals_instance_mul.append(goals_instance_i)
                goals_probs_mul.append(goals_probs_i)
            return goals_instance_mul, goals_probs_mul
    # ------------------------------------------------------------------------------------

    if number_answers == 1:
        goals_probs = get_predicate_probabilities(response_goals, response_format_goals)
        return [goals_instance], [goals_probs]
    else:
        goals_instance_mul = []
        goals_probs_mul = []
        for i in range(number_answers):
            goals_probs_i = get_predicate_probabilities(response_goals, response_format_goals, iteration_number=i)
            goals_instance_i = response_goals.choices[i].message.parsed
            goals_instance_mul.append(goals_instance_i)
            goals_probs_mul.append(goals_probs_i)
        return goals_instance_mul, goals_probs_mul


def main_pipeline(
    image_path: str,
    domain_name: str,
    action: str,
    main_dataset_path: str = "datasets/master_dataset.json",
    next_state_dataset_path: str = "datasets/next_state_dataset.json",
    num_examples: int = 1,
    problem_id = -1
):
    """
    Main pipeline to retrieve objects, predicates, and predict next state.
    """
    main_dataset = load_dataset(main_dataset_path)
    objects, object_probs = retrieve_objects(
        image_path=image_path,
        domain_name=domain_name,
        dataset=main_dataset,
        num_examples=num_examples,
        dataset_path=main_dataset_path, 
        number_answers = 1,
        problem_id=problem_id
    )
    
    if objects is None:
        print("Object retrieval failed. Aborting pipeline.")
        return None, None, None
    
    predicates, predicate_probs = retrieve_predicates(
        image_path=image_path,
        domain_name=domain_name,
        objects=objects,
        dataset=main_dataset,
        num_examples=num_examples,
        dataset_path=main_dataset_path,
        problem_id=problem_id
    )
    if predicates is None:
        print("Predicate retrieval failed. Aborting pipeline.")
        return objects, None, None
    if isinstance(predicates, list):
        predicates = predicates[0]
    next_state_dataset = load_dataset(next_state_dataset_path)
    current_state = {
        "grounded_predicates": predicates.dict().get("grounded_predicates", [])
    }
    next_state = predict_next_state(
        action=action,
        current_state=current_state,
        objects=objects,
        domain_name=domain_name,
        dataset=next_state_dataset,
        num_examples=num_examples,
        dataset_path=next_state_dataset_path,
        problem_id=problem_id
    )
    
    return objects, predicates, next_state

def find_problem_num(env_name,dataset,dataset_path):
    """
    Attempt to find any matching problem for this domain in the dataset.
    """
    domain_key = env_name.replace("PDDLEnv", "").replace("-v0", "").lower()
    problem_nums = []
    for i, prob in enumerate(dataset["problems"]):
        if "domain_name" not in prob:
            # Fall back to problem_name if domain_name is missing
            domain_name = prob.get("problem_name", "")
        else:
            domain_name = prob["domain_name"]
        if domain_name == domain_key:
            problem_nums.append(i)
    if not problem_nums:
        raise ValueError(f"No problems found for domain {domain_key} in dataset {dataset_path}")
    return random.choice(problem_nums)


if __name__ == "__main__":
    # raise Exception("This script is not meant to be run directly.")
    image_path = "./experiments/data/pddlgym_grounding/blocks/observations/pddlgym_blocks_problem_1_sample_1.jpg"
    domain_name = "blocks"
    action = "move_block_a_to_block_b"
    dataset_path = "./experiments/data/datasets/pddlGYM_dataset_grounding.json"
    num_examples = 5
    raise Exception("This script is not meant to be run directly. Use simulator")
    next_state_dataset_path = "./experiments/data/datasets/pddlGYM_dataset_grounding.json"
    problem_id = find_problem_num(domain_name,load_dataset(dataset_path),dataset_path)
    objects, predicates, next_state = main_pipeline(
        image_path=image_path,
        domain_name=domain_name,
        action=action,
        main_dataset_path=dataset_path,
        next_state_dataset_path=next_state_dataset_path,
        num_examples=num_examples,
        problem_id=problem_id
    )
    
    print("Objects:", objects)
    print("Predicates:", predicates)
    print("Predicted Next State:", next_state)
