# %% [markdown]
# Imports

# %%

import os
from pddlgym.utils import run_demo
from pddlgym.utils import run_demo
import pddlgym
import matplotlib
import imageio
from pddl import parse_problem
from .chatgpt_utils import parse_domain
from pddl.logic import Predicate, constants, variables
from pddl.core import Domain, Problem
from pddl.action import Action
import pddl 


USE_LOGPROBS = os.getenv("USE_LOGPROBS", "false").lower() == "true"
DEFAULT_TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "20" if USE_LOGPROBS else "0"))


def _resolve_type_hierarchy(types_available_classes_mapping, domain_types):
    """Recursively resolve multi-level type hierarchies for PDDL domains.

    Populates `types_available_classes_mapping` so that each parent type
    contains all objects from all its descendant types. Also ensures the
    implicit PDDL root type 'object' is populated with all objects.
    """
    resolved = set()
    def _collect(type_name):
        if type_name in resolved:
            return types_available_classes_mapping.get(type_name, [])
        resolved.add(type_name)
        if type_name not in types_available_classes_mapping:
            types_available_classes_mapping[type_name] = []
        if type_name in domain_types:
            for subclass in domain_types[type_name]:
                for obj in _collect(subclass):
                    if obj not in types_available_classes_mapping[type_name]:
                        types_available_classes_mapping[type_name].append(obj)
        return types_available_classes_mapping[type_name]

    for superclass in domain_types.keys():
        if superclass is not None and "None" not in str(superclass):
            _collect(superclass)

    # Ensure 'object' (PDDL root type) includes ALL objects.
    # In PDDL, every object is implicitly of type 'object', so the root type
    # must always contain the union of all objects regardless of explicit typing.
    all_obj_names = []
    for names in types_available_classes_mapping.values():
        for n in names:
            if n not in all_obj_names:
                all_obj_names.append(n)
    if all_obj_names:
        types_available_classes_mapping["object"] = all_obj_names


def init_gym_env(
    problem_name : str,
    problem_index : int,
    ):

    # need objects and state
    # Create the environment and two images one before one after an random action
    env = pddlgym.make("PDDLEnv{}-v0".format(problem_name.capitalize()))
    env.fix_problem_index(problem_index)

    # Reset the environment to get the initial observation and debug information
    obs, debug_info = env.reset()
    
    return env, obs, debug_info

def get_original_objects_and_atoms_from_obs(obs, domain_file_path):
    types_available_classes_mapping = {}
    # Get the Object schema from the PDDL domain file
    object_schema, list_types = get_object_schema_from_pddl(domain_file_path)
    
    all_objects_original = []
    for object_pddl in list(obs.objects):
        str_type = str(object_pddl.var_type)
        str_name = str(object_pddl.name)
        all_objects_original.append({"name": str_name, "type": str_type})
        
    all_objects_original_json = {
        "objects": all_objects_original
    }

    # Parse the JSON data using the ObjectList model
    all_objects_original_instance = object_schema(**all_objects_original_json)
    
    for object_instance in all_objects_original_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(object_instance.name)


    # Ensure 'object' (PDDL root type) includes ALL objects
    if "object" not in types_available_classes_mapping:
        all_obj_names = []
        for names in types_available_classes_mapping.values():
            for n in names:
                if n not in all_obj_names:
                    all_obj_names.append(n)
        if all_obj_names:
            types_available_classes_mapping["object"] = all_obj_names

    # Get domain wrapper and predicates from the file
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    domain_wrapper = PDDLDomainWrapper(domain_file_path)

    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
        
    # Transform predicates to a simplified dictionary
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}

    # Structure predicates with variable naming
    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        missing_classes = all_classes_used - classes_with_objects
        if not missing_classes:
            predicates_structured[pred_name] = dict_how_are_they_named        
        else:
            print(f"No object for classes: {pred_name} because objects of classes {missing_classes} are missing")
    
    # Create dynamic models
    predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping)

    # Extract atoms from observation
    all_atoms_original = []
    for atom in obs.literals:
        predicate_name = atom.predicate.name.capitalize()
        next_char = "x"

        atom_dict = {"predicate_type": predicate_name}
        for variable_obj in atom.variables:
            var_name = variable_obj.name
            atom_dict[next_char] = {"name": var_name}
            next_char = next_letter(next_char)
        all_atoms_original.append(atom_dict)
        
    grounded_atoms_original_json_data = {
        "grounded_predicates": all_atoms_original
    }
    grounded_atoms_original_instance = Atoms(**grounded_atoms_original_json_data)
    
    return all_objects_original_instance, grounded_atoms_original_instance, predicate_classes, Atoms, generated_classes, predicates_structured



def make_messages_image(prompt, 
                        image_base64,
                        low_resolution = False,
                        ):
    if low_resolution:
        message = {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail":"low"
                    }
                }
                ]
                }
    else:   
        message = {"role": "user", "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                }
            }
            ]
            }
    return message

def make_message_text(prompt):
    message = {"role": "user", "content": prompt}
    return message

def get_structured_output_parsed(
    client,
    messages,
    schema_response,
    model_name
    ):

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "response_format": schema_response,
    }
    if USE_LOGPROBS and DEFAULT_TOP_LOGPROBS > 0:
        request_kwargs.update({
            "logprobs": True,
            "top_logprobs": DEFAULT_TOP_LOGPROBS,
        })

    object_completion = client.beta.chat.completions.parse(**request_kwargs)
    	
    object_list_instance  = object_completion.choices[0].message.parsed
    return object_completion, object_list_instance

# def get_atom_schema_from_chatgpt_objects(
#     object_list_instance,
#     pddl_domain_file_path,
#     types_available_classes_mapping
#     ):

#     for object_instance in object_list_instance.objects:
#         if object_instance.type not in types_available_classes_mapping.keys():
#             types_available_classes_mapping[object_instance.type] = []
#         types_available_classes_mapping[object_instance.type].append(object_instance.name)        

#     classes_with_objects = set(list(types_available_classes_mapping.keys()))
#     #get domain wrapper and domains form the file:
#     domain_wrapper = PDDLDomainWrapper(pddl_domain_file_path)

#     predicates = {}
#     for gen_pred in list(domain_wrapper.predicates):
#         pred = {}
#         pred['name'] = gen_pred.name
#         pred['arity'] = gen_pred.arity
#         pred['types'] = [t.type_tags for t in gen_pred.terms]
#         predicates[gen_pred.name] = pred
        
#     #turn into dict with list:
#     def transform_dict(original_dict):
#         simplified_dict = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in original_dict.items()}
#         return simplified_dict

#     predicates = transform_dict(predicates)

#     #enum for each one with objects..

#     predicates_structured = {}
#     for pred_name, pred in predicates.items():
#         next_char = 'x'
#         dict_how_are_they_named = {}
#         for type_name in pred:
#             dict_type_name = {str("" + next_char):type_name}
#             dict_how_are_they_named = {**dict_how_are_they_named, **dict_type_name}
#             next_char = next_letter(next_char)
#         all_classes_used = set([dict_how_are_they_named[key] for key in dict_how_are_they_named.keys()])
#         if all_classes_used-classes_with_objects == set():
#             predicates_structured[pred_name] = dict_how_are_they_named        
#         else:
#             print("No object for classes: ",pred_name,"because an object of class", all_classes_used-classes_with_objects,"is missing")
            
#     #get the predicate classes
#     predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping)
     
#     return Atoms

from pddl.formatter import domain_to_string, problem_to_string
from pddl.requirements import Requirements
from typing import List, Dict, Any, Set, Collection
from dataclasses import dataclass
from typing import List, Literal
from pydantic import create_model, BaseModel, Field, ValidationError
from enum import Enum
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import requests
from typing import List, Union, Literal, Dict, Any


def _rebuild_object_schema(all_types):
    """Rebuild object schema with an expanded set of types."""
    list_types = tuple(sorted(all_types))
    class Object(BaseModel):
        name: str
        type: Literal[list_types] = Field(..., description="Must be one of the valid types")
    class ObjectList(BaseModel):
        objects: List[Object]
    return ObjectList, list_types
from PIL import Image
import time 
import pddlgym
import matplotlib
from .general_utils import load_env, next_letter
from .chatgpt_utils import PDDLDomainWrapper, PDDLProblemWrapper,_print_predicates_with_types,get_object_schema_from_pddl, get_object_schema_from_pddl, create_classes_from_dict, create_dynamic_models,encode_image_base64_from_url,encode_image_base64_from_local, encode_image_base64
# %% [markdown]
# Pddl Gym Handling

# %%
#pddl get video and image
def demo_random(env_name, render=True, problem_index=0, verbose=True):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    env.fix_problem_index(problem_index)
    policy = lambda s : env.action_space.sample(s)
    video_path = "./images/{}_random_demo.mp4".format(env_name)
    import os
    print("cwd: ", os.getcwd())
    print("Video saved to: {}".format(video_path))
    run_demo(env, policy, render=True, verbose=True, seed=0,
             video_path=video_path)

def make_image_from_domain(domain_name):
    env = pddlgym.make("PDDLEnv{}-v0".format(domain_name.capitalize()))
    obs, debug_info = env.reset()
    img = env.render()
    img_path = "images/{}.png".format(domain_name)
    imageio.imsave(img_path, img)
    return img_path



# %% [markdown]
# Available Domains

# %%
all_domains = ['quantifiedblocks3',
 'generated_blocks',
 'blocks_operator_actions',
 'manyblocksnopiles',
 'doors',
 'searchandrescue_level3',
 'blocks_operator_actions_test',
 'glibblocks',
 'rearrangement-notyping',
 'blocks_medium_test',
 'searchandrescue_level6',
 'manygrid_test',
 'searchandrescue_level7_test',
 'snake_test',
 'manyferry_test',
 'manytireworld',
 'hiking_test',
 'newspapers',
 'manytireworld_test',
 'manyblockssmallpiles_test',
 'onearmedgripper_test',
 'derivedblocks',
 'rearrangement_test',
 'visit_all',
 'elevator',
 'depot_test',
 'searchandrescue_level4',
 'fridge',
 'generated_blocks_test',
 'glibblocks_test',
 'tyreworld',
 'quantum_circuit',
 'conditionalblocks_test',
 'explodingblocks_test',
 'onearmedgripper',
 'explodingblocks',
 'navigation2',
 'manyblockssmallpiles',
 'casino',
 'toomanyblocks',
 'quantifiedblocks2',
 'logistics',
 'blocks_subgoal_problems_test',
 'optgripper',
 'casino_test',
 'spannerlearning',
 'tireworld',
 'easynewspapers_test',
 'manygripper_test',
 'lifelong_blocks6',
 'manymiconic',
 'quantifiedblocks',
 'minecraft',
 'easynewspapers',
 'searchandrescue_level6_test',
 'trapnewspapers',
 'footwear_test',
 'travelhard',
 'spannerlearning_test',
 'blocks_medium',
 'dynamic_action_space_same_obj',
 'manyexplodingblockssmallpiles_test',
 'lifelong_tiny_gripper',
 'manymiconic_test',
 'equalityblocks',
 'navigation1',
 'baking_test',
 'hanoi_test',
 'hanoi_operator_actions',
 'manylogistics',
 'blocks_pile_of_two_test',
 'conditionalferry',
 'manyexplodingblockssmallpiles',
 'travel_test',
 'orbit',
 'navigation4',
 'glibdoors',
 'tsp_operator_actions_test',
 'manyblocksnopiles_test',
 'conditionalblocks',
 'searchandrescue_level4_test',
 'easyblocks',
 'tireworld_test',
 'meetpass',
 'trapnewspapers_test',
 'glibdoors_test',
 'hanoi',
 'lifelong_blocks6_test',
 'searchandrescue_level5',
 'searchandrescue_level1',
 'tinyonearmedgripper',
 'blocks_test',
 'quantifiedblocks_test',
 'meetpass_test',
 'searchandrescue_level3_test',
 'travel',
 'tsp_operator_actions',
 'blocks_subgoal_problems_train',
 'navigation10',
 'navigation5',
 'glibrearrangement_test',
 'searchandrescue_level2',
 'slidetile',
 'easygripper',
 'lifelong_tiny_grid',
 'river',
 'navigation7',
 'navigation',
 'navigation8',
 'conditionalferry_test',
 'movie_test',
 'snake',
 'travelhard_test',
 'slidetile_test',
 'newspapers_test',
 'hiking',
 'fridge_test',
 'manylogistics_test',
 'tinyonearmedgripper_test',
 'ferry',
 'miconic',
 'unfactored_manygripper',
 'elevator_test',
 'footwear',
 'searchandrescue_level5_test',
 'hanoi_operator_actions_test',
 'blocks',
 'searchandrescue_level2_test',
 'glibrearrangement',
 'rearrangement',
 'easyblocks_test',
 'navigation3',
 'navigation6',
 'lifelong_tiny_gripper_test',
 'river_test',
 'depot',
 'quantifiedblocks2_test',
 'movie',
 'ferry_test',
 'manygrid',
 'parking',
 'navigation9',
 'searchandrescue_level7',
 'manyferry',
 'gripper',
 'tsp_test',
 'maze_test',
 'baking',
 'searchandrescue_level1_test',
 'easygripper_test',
 'doors_test',
 'equalityblocks2',
 'gripper_test',
 'maze',
 'minecraft_test',
 'tsp',
 'manygripper',
 'rearrangement-notyping_test']

# %%
# #get the working domains, if needed fir to get alll
# all_domains_working = []
# real_names = {}
# for domain in all_domains:
#     try: 
#         new_name = domain.replace("_level", "Level")
#         new_name = new_name.replace("_test", "Test")
#         new_name = new_name.replace("test", "Test")
#         real_names[new_name] = domain
#         make_image_from_domain(new_name)
#         all_domains_working.append(new_name)
#     except Exception as e:
#         print("Error: ", e)
#         print("Domain: ", new_name)
#         continue
#     except KeyboardInterrupt:
#         break

# %%
all_domains_working = ['quantifiedblocks3',
 'generated_blocks',
 'blocks_operator_actions',
 'manyblocksnopiles',
 'doors',
 'glibblocks',
 'manytireworld',
 'visit_all',
 'explodingblocks',
 'navigation2',
 'manyblockssmallpiles',
 'toomanyblocks',
 'quantifiedblocks2',
 'tireworld',
 'lifelong_blocks6',
 'quantifiedblocks',
 'minecraft',
 'blocks_medium',
 'navigation1',
 'hanoi_operator_actions',
 'manyexplodingblockssmallpiles',
 'navigation4',
 'glibdoors',
 'conditionalblocks',
 'easyblocks',
 'hanoi',
 'tsp_operator_actions',
 'navigation10',
 'navigation5',
 'slidetile',
 'navigation7',
 'navigation8',
 'blocks',
 'glibrearrangement',
 'rearrangement',
 'navigation3',
 'navigation6',
 'navigation9',
 'maze',
 'tsp']
# %%
# selected_domains  = [
#  'blocks_operator_actions',
#  'doors',
#  'visit_all',
#  'minecraft',
#  'navigation1',
#  'hanoi_operator_actions',
#  'glibdoors',
#  'hanoi',#numbers/ colors
#  'slidetile',#not good numbers but ok
#  'glibrearrangement',
#  'rearrangement',
#  'maze',
#  'tsp']#need numbers
# %%
selected_domains_old  = [
 'blocks_operator_actions',
 'doors',
 'visit_all',
 'minecraft',
 'navigation1',
 'hanoi_operator_actions',
 'glibdoors',
 'hanoi',#numbers/ colors
 'slidetile',#not good numbers but ok
 'glibrearrangement',
 'rearrangement',
 'maze',
 'tsp']#need numbers

# %%
selected_domains  = [
 'blocks_operator_actions_encoded',
 'blocks_operator_actions',
 'doors',#actions in the predicates # need description for keys
 'visit_all',
 'minecraft',#actions in the predicates
 'rearrangement',
 'tsp',
#  'doors_encoded',#actions in the predicates # need description for keys
 'visit_all_encoded',
#  'minecraft_encoded',#actions in the predicates
#  'rearrangement_encoded',
 'tsp_encoded' #TODO ADD
 ]#need numbers


def init_gym_env(
    problem_name : str,
    problem_index : int,
    ):

    # need objects and state
    # Create the environment and two images one before one after an random action
    env = pddlgym.make("PDDLEnv{}-v0".format(problem_name.capitalize()))
    env.fix_problem_index(problem_index)

    # Reset the environment to get the initial observation and debug information
    obs, debug_info = env.reset()
    
    return env, obs, debug_info

def get_original_objects_and_atoms_from_obs(obs, domain_file_path):
    types_available_classes_mapping = {}
    # Get the Object schema from the PDDL domain file
    object_schema, list_types = get_object_schema_from_pddl(domain_file_path)
    
    all_objects_original = []
    for object_pddl in list(obs.objects):
        str_type = str(object_pddl.var_type)
        str_name = str(object_pddl.name)
        all_objects_original.append({"name": str_name, "type": str_type})
        
    all_objects_original_json = {
        "objects": all_objects_original
    }

    # Parse the JSON data using the ObjectList model
    all_objects_original_instance = object_schema(**all_objects_original_json)
    
    for object_instance in all_objects_original_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(object_instance.name)


    # Ensure 'object' (PDDL root type) includes ALL objects
    if "object" not in types_available_classes_mapping:
        all_obj_names = []
        for names in types_available_classes_mapping.values():
            for n in names:
                if n not in all_obj_names:
                    all_obj_names.append(n)
        if all_obj_names:
            types_available_classes_mapping["object"] = all_obj_names

    # Get domain wrapper and predicates from the file
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    domain_wrapper = PDDLDomainWrapper(domain_file_path)

    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
        
    # Transform predicates to a simplified dictionary
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}

    # Structure predicates with variable naming
    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        missing_classes = all_classes_used - classes_with_objects
        if not missing_classes:
            predicates_structured[pred_name] = dict_how_are_they_named        
        else:
            print(f"No object for classes: {pred_name} because objects of classes {missing_classes} are missing")
    
    # Create dynamic models
    predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping)

    # Extract atoms from observation
    all_atoms_original = []
    for atom in obs.literals:
        predicate_name = atom.predicate.name.capitalize()
        next_char = "x"

        atom_dict = {"predicate_type": predicate_name}
        for variable_obj in atom.variables:
            var_name = variable_obj.name
            atom_dict[next_char] = {"name": var_name}
            next_char = next_letter(next_char)
        all_atoms_original.append(atom_dict)
        
    grounded_atoms_original_json_data = {
        "grounded_predicates": all_atoms_original
    }
    grounded_atoms_original_instance = Atoms(**grounded_atoms_original_json_data)
    
    return all_objects_original_instance, grounded_atoms_original_instance, predicate_classes, Atoms, generated_classes, predicates_structured



def make_messages_image(prompt, 
                        image_base64,
                        low_resolution = False,
                        ):
    if low_resolution:
        message = {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail":"low"
                    }
                }
                ]
                }
    else:   
        message = {"role": "user", "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                }
            }
            ]
            }
    return message

def make_message_text(prompt):
    message = {"role": "user", "content": prompt}
    return message

def get_structured_output_parsed(
    client,
    messages,
    schema_response,
    model_name
    ):

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "response_format": schema_response,
    }
    if USE_LOGPROBS and DEFAULT_TOP_LOGPROBS > 0:
        request_kwargs.update({
            "logprobs": True,
            "top_logprobs": DEFAULT_TOP_LOGPROBS,
        })

    object_completion = client.beta.chat.completions.parse(**request_kwargs)
	
    object_list_instance  = object_completion.choices[0].message.parsed
    return object_completion, object_list_instance

# def get_atom_schema_from_chatgpt_objects(
#     object_list_instance,
#     pddl_domain_file_path,
#     types_available_classes_mapping
#     ):

#     for object_instance in object_list_instance.objects:
#         if object_instance.type not in types_available_classes_mapping.keys():
#             types_available_classes_mapping[object_instance.type] = []
#         types_available_classes_mapping[object_instance.type].append(object_instance.name)        

#     classes_with_objects = set(list(types_available_classes_mapping.keys()))
#     #get domain wrapper and domains form the file:
#     domain_wrapper = PDDLDomainWrapper(pddl_domain_file_path)

#     predicates = {}
#     for gen_pred in list(domain_wrapper.predicates):
#         pred = {}
#         pred['name'] = gen_pred.name
#         pred['arity'] = gen_pred.arity
#         pred['types'] = [t.type_tags for t in gen_pred.terms]
#         predicates[gen_pred.name] = pred
        
#     #turn into dict with list:
#     def transform_dict(original_dict):
#         simplified_dict = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in original_dict.items()}
#         return simplified_dict

#     predicates = transform_dict(predicates)

#     #enum for each one with objects..

#     predicates_structured = {}
#     for pred_name, pred in predicates.items():
#         next_char = 'x'
#         dict_how_are_they_named = {}
#         for type_name in pred:
#             dict_type_name = {str("" + next_char):type_name}
#             dict_how_are_they_named = {**dict_how_are_they_named, **dict_type_name}
#             next_char = next_letter(next_char)
#         all_classes_used = set([dict_how_are_they_named[key] for key in dict_how_are_they_named.keys()])
#         if all_classes_used-classes_with_objects == set():
#             predicates_structured[pred_name] = dict_how_are_they_named        
#         else:
#             print("No object for classes: ",pred_name,"because an object of class", all_classes_used-classes_with_objects,"is missing")
            
#     #get the predicate classes
#     predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping)
     
#     return Atoms
def get_atom_schema_from_chatgpt_objects(object_list_instance: Any, pddl_domain_file_path: str, include_negation: bool = False) -> Any:
    # Parse domain and problem
    domain = parse_domain(pddl_domain_file_path)

    # Creating the inverted dictionary
    domain_types = {}
    for key, value in domain.types.items():
        if value not in domain_types:
            domain_types[value] = []
        domain_types[value].append(key)
    # Extract objects and types from the problem
    # Assuming problem.objects is a dict: {object_name: object_type}
    all_objects_original = []

    for obj_thing in list(object_list_instance.objects):
        all_objects_original.append({"name": obj_thing.name, "type": str(obj_thing.type)})

    all_objects_original_json = {
        "objects": all_objects_original
    }

    
    # Get the object schema from the PDDL domain file
    object_schema, list_types = get_object_schema_from_pddl(pddl_domain_file_path)

    # Rebuild schema if objects use types not in the domain
    gt_types = {o["type"] for o in all_objects_original}
    extra_types = gt_types - set(list_types)
    if extra_types:
        all_types = tuple(sorted(set(list_types) | gt_types))
        object_schema, list_types = _rebuild_object_schema(all_types)

    all_objects_original_instance = object_schema(**all_objects_original_json)

    # Map types to objects
    types_available_classes_mapping = {}
    for object_instance in all_objects_original_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(object_instance.name)

    _resolve_type_hierarchy(types_available_classes_mapping, domain_types)

    # Load the domain wrapper to access predicates
    domain_wrapper = PDDLDomainWrapper(pddl_domain_file_path)

    # Extract predicate type information
    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
    # Transform predicates to a simplified dictionary: {predicate_name: [type1, type2, ...]}
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}
    # Structure predicates with variable naming
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        missing_classes = all_classes_used - classes_with_objects
        if not missing_classes:
            predicates_structured[pred_name] = dict_how_are_they_named
        else:
            print(f"No object for classes in predicate '{pred_name}' because objects of classes {missing_classes} are missing")
    
    # Create dynamic models for atoms and predicates
    predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping, include_negation=include_negation)
    return  Atoms

def get_original_objects_and_atoms_from_problem(domain_file_path, problem_file_path):
    # Parse domain and problem
    domain = parse_domain(domain_file_path)
    problem = parse_problem(problem_file_path)

    # Creating the inverted dictionary
    domain_types = {}
    for key, value in domain.types.items():
        if value not in domain_types:
            domain_types[value] = []
        domain_types[value].append(key)
    # Extract objects and types from the problem
    # Assuming problem.objects is a dict: {object_name: object_type}
    all_objects_original = []

    for obj_thing in list(problem.objects):
        all_objects_original.append({"name": obj_thing.name, "type": str(obj_thing.type_tag)})

    all_objects_original_json = {
        "objects": all_objects_original
    }

    
    # Get the object schema from the PDDL domain file
    object_schema, list_types = get_object_schema_from_pddl(domain_file_path)
    all_objects_original_instance = object_schema(**all_objects_original_json)
    
    # Map types to objects
    types_available_classes_mapping = {}
    for object_instance in all_objects_original_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(object_instance.name)

    _resolve_type_hierarchy(types_available_classes_mapping, domain_types)

    # Load the domain wrapper to access predicates
    domain_wrapper = PDDLDomainWrapper(domain_file_path)

    # Extract predicate type information
    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
    # Transform predicates to a simplified dictionary: {predicate_name: [type1, type2, ...]}
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}
    # Structure predicates with variable naming
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        missing_classes = all_classes_used - classes_with_objects
        if not missing_classes:
            predicates_structured[pred_name] = dict_how_are_they_named
        else:
            print(f"No object for classes in predicate '{pred_name}' because objects of classes {missing_classes} are missing")
    
    # Create dynamic models for atoms and predicates
    predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping)

    # Extract atoms from the initial state of the problem
    # Assuming problem.init is a collection of ground atoms (with a predicate and variables)
    all_atoms_original = []
    for atom in problem.init:
        # atom.predicate.name and atom.variables could differ based on your pddl library
        predicate_name = atom.name.capitalize()
        next_char = "x"
        atom_dict = {"predicate_type": predicate_name}
        for variable_obj in atom.terms:
            # variable_obj.name should give the object's name
            var_name = variable_obj.name
            atom_dict[next_char] = {"name": str(var_name)}
            next_char = next_letter(next_char)
        all_atoms_original.append(atom_dict)

    grounded_atoms_original_json_data = {
        "grounded_predicates": all_atoms_original
    }

    grounded_atoms_original_instance = Atoms(**grounded_atoms_original_json_data)

    return (all_objects_original_instance, grounded_atoms_original_instance, 
            predicate_classes, Atoms, generated_classes, predicates_structured)

# %%

def get_goal_from_problem(domain_file_path, problem_file_path):
    # Parse domain and problem
    domain = parse_domain(domain_file_path)
    problem = parse_problem(problem_file_path)

    # Creating the inverted dictionary
    domain_types = {}
    for key, value in domain.types.items():
        if value not in domain_types:
            domain_types[value] = []
        domain_types[value].append(key)
    # Extract objects and types from the problem
    # Assuming problem.objects is a dict: {object_name: object_type}
    all_objects_original = []

    for obj_thing in list(problem.objects):
        all_objects_original.append({"name": obj_thing.name, "type": str(obj_thing.type_tag)})

    all_objects_original_json = {
        "objects": all_objects_original
    }

    
    # Get the object schema from the PDDL domain file
    object_schema, list_types = get_object_schema_from_pddl(domain_file_path)
    all_objects_original_instance = object_schema(**all_objects_original_json)

    # Map types to objects
    types_available_classes_mapping = {}
    for object_instance in all_objects_original_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(object_instance.name)

    _resolve_type_hierarchy(types_available_classes_mapping, domain_types)

    # Load the domain wrapper to access predicates
    domain_wrapper = PDDLDomainWrapper(domain_file_path)

    # Extract predicate type information
    predicates = {}
    for gen_pred in list(domain_wrapper.predicates):
        pred = {}
        pred['name'] = gen_pred.name
        pred['arity'] = gen_pred.arity
        pred['types'] = [t.type_tags for t in gen_pred.terms]
        predicates[gen_pred.name] = pred
    # Transform predicates to a simplified dictionary: {predicate_name: [type1, type2, ...]}
    predicates = {k: [list(t)[0] if t else "object" for t in v['types']] for k, v in predicates.items()}
    # Structure predicates with variable naming
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    predicates_structured = {}
    for pred_name, pred_types in predicates.items():
        next_char = 'x'
        dict_how_are_they_named = {}
        for type_name in pred_types:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        missing_classes = all_classes_used - classes_with_objects
        if not missing_classes:
            predicates_structured[pred_name] = dict_how_are_they_named
        else:
            print(f"No object for classes in predicate '{pred_name}' because objects of classes {missing_classes} are missing")
    
    # Create dynamic models for atoms and predicates (with negation support for goals)
    predicate_classes, Atoms, generated_classes = create_dynamic_models(predicates_structured, types_available_classes_mapping, include_negation=True)

    # Extract goal atoms from the problem, including negated ones
    all_atoms_original = []
    if isinstance(problem.goal, pddl.logic.base.And):
        goal_atoms = problem.goal.operands
    elif isinstance(problem.goal, pddl.logic.predicates.Predicate):
        goal_atoms = [problem.goal]
    else:
        raise Exception("Debug this")

    for atom in goal_atoms:
        # Handle negated goal atoms: (not (predicate args...))
        is_negated = False
        actual_atom = atom
        if isinstance(atom, pddl.logic.base.Not):
            is_negated = True
            # The negated formula wraps a single atom
            operands = list(atom.operands) if hasattr(atom, 'operands') else [atom.argument]
            if len(operands) != 1 or not isinstance(operands[0], pddl.logic.predicates.Predicate):
                raise Exception(f"Unsupported negated goal formula: {atom}")
            actual_atom = operands[0]

        predicate_name = actual_atom.name.capitalize()
        next_char = "x"
        atom_dict = {"predicate_type": predicate_name, "is_negated": is_negated}
        for variable_obj in actual_atom.terms:
            var_name = variable_obj.name
            atom_dict[next_char] = {"name": str(var_name)}
            next_char = next_letter(next_char)
        all_atoms_original.append(atom_dict)

    grounded_atoms_original_json_data = {
        "grounded_predicates": all_atoms_original
    }
    

    grounded_atoms_original_instance = Atoms(**grounded_atoms_original_json_data)
    
    return grounded_atoms_original_instance
