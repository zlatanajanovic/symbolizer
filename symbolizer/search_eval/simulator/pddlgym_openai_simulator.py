import random
import itertools
import logging
from enum import Enum
from typing import Any, List, Dict, Set, Optional
from pddlgym import make
from pddlgym.inference import check_goal
from pddlgym.core import get_successor_state
from pddlgym.structs import State, Literal, Predicate, Type, TypedEntity, LiteralConjunction
#print the cwd
# Add 'powbs' directory to sys.path
import os
import sys
import imageio
import hashlib
from PIL import Image
import numpy as np
import ast

# Set up logging
logger = logging.getLogger(__name__)


class SSRFailurePolicy(Enum):
    """Policy for handling SSR failures during observation.
    
    ONLY RAISE is supported. All fallback modes have been removed.
    If SSR fails, the experiment fails — no silent fallbacks.
    """
    RAISE = "raise"           # Raise an exception on SSR failure
    # REMOVED: FALLBACK_GT, FALLBACK_EMPTY, WARN_AND_CONTINUE
    # All fallbacks removed per project policy: if something fails, it fails.


class SSRObjectRetrievalError(Exception):
    """Exception raised when SSR object retrieval fails."""
    pass


class SSRPredicateRetrievalError(Exception):
    """Exception raised when SSR predicate retrieval fails."""
    pass

# Add the project root directory to sys.path
# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
#add the ssr_parsing directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ssr_parsing'))
from .pddlgym_simulator import PDDLGymSimulator

from ssr_parsing.ssr_parsing import (
    retrieve_objects,
    retrieve_predicates,
    predict_next_state as ssr_predict_next_state,
    select_best_action as ssr_select_best_action,
    score_state_heuristic as ssr_score_state_heuristic,
    get_simplified_object_probs,
    get_simplified_pred_probs,
    load_dataset,
    retrieve_goal
)

# VLM client init is deferred to first simulator construction so that
# importing search/eval code never requires API credentials.
from ssr_parsing.run_eval_vlm import reinitialize_client as _reinit_vlm

def _ensure_vlm_client():
    from ssr_parsing import run_eval_vlm as _rev
    if _rev.client is None:
        _reinit_vlm()

class PDDLGymSimulatorOpenai:
    """
    Simulator implementation using PDDL Gym with the SSR parsing pipeline.
    This simulator integrates image-based retrieval of objects/predicates,
    plus optional noise or threshold-based filtering for observations.
    """

    def __init__(
        self,
        env_name: str,
        dataset_path: str,
        image_folder_path: str,
        next_state_dataset_path: str,
        num_examples = 1,
        problem_index: int = 0,
        object_threshold_propability = 0.0,
        use_original_objects = False,
        chosen_goal = None,
        step_object_checking = 5,
        ssr_failure_policy: str = "raise"
    ):
        """
        Initialize the PDDLGym simulator with SSR-based state observation.

        Args:
            env_name: The name of the PDDL Gym environment.
            dataset_path: Path to the dataset (used by SSR).
            image_folder_path: Directory where images for SSR are stored.
            next_state_dataset_path: Path to the next_state dataset for SSR.
            num_examples: Number of few-shot examples to use in SSR calls.
            problem_index: Which problem index to load in the environment.
            object_threshold_propability: Probability threshold for filtering out low-probability objects.
            use_original_objects: Whether to keep the environment's original objects or rely on SSR.
            chosen_goal: The goal to be used for the environment.
            step_object_checking: The number of steps to check for objects. If 0, it will check every step. Default is 5.
            ssr_failure_policy: How to handle SSR failures. Only 'raise' is supported.
        """
        _ensure_vlm_client()
        self.env_name = env_name
        self.dataset_path = dataset_path
        self.image_folder_path = image_folder_path
        self.next_state_dataset_path = next_state_dataset_path
        self.num_examples = num_examples
        self.problem_index = problem_index
        self.ssr_failure_policy = SSRFailurePolicy(ssr_failure_policy)
        self.object_threshold_propability = object_threshold_propability
        self.use_original_objects = use_original_objects
        self.all_objects_probs_list = None
        self.retrieved_objs_list = None
        # Create and fix the environment problem
        self.env = make(env_name)
        self.env.fix_problem_index(problem_index)
        self.chosen_goal = chosen_goal
        # Internal simulator state
        self.state = None
        self.domain = self.env.domain
        self.objects = None
        self.retrieved_objs = None
        # All possible atoms (may be used for reference or other logic)
        self.all_atoms: Set[Literal] = set()

        # Dictionary to store per-atom probabilities (from observations)
        self.atom_probabilities: Dict[Literal, float] = {}

        # Dictionary to store per-atom probabilities for predictions
        self.atom_probabilities_prediction: Dict[Literal, float] = {}

        self.dataset = load_dataset(self.dataset_path)
        self.dataset_pred = load_dataset(self.next_state_dataset_path)

        self.problem_num = self.find_problem_num()

        self.step_without_checking = 0
        self.num_steps_object_checking = step_object_checking
        
    def reset(self) -> State:
        """
        Reset the simulator to the initial state from PDDLGym.

        Returns:
            The initial state.
        """
        self.state, _ = self.env.reset()
        if not self.chosen_goal:
            # self.state.goal = self.state.goal
            self.chosen_goal = self.state.goal

        # If already a PDDLGym goal object (Literal or LiteralConjunction), use it directly
        if isinstance(self.chosen_goal, (Literal, LiteralConjunction)):
            # Already a proper goal object, no parsing needed
            pass
        elif isinstance(self.chosen_goal, str):
            # String goal needs parsing
            if "[" in str(self.chosen_goal):
                self.chosen_goal = ast.literal_eval(self.chosen_goal)
        if isinstance(self.chosen_goal,str):
            for lit in self.state.goal.literals:
                if str(lit) == self.chosen_goal:
                    self.chosen_goal = lit
                    break
            if isinstance(self.chosen_goal, str):
                print(self.state.goal.literals)
                raise ValueError(f"Goal {self.chosen_goal} not found in the goal literals")
            
        if isinstance(self.chosen_goal, list):
            # If the goal is a list, we need to convert it to a set of literals
            chosen_goal_conjunction = []
            for lit in self.state.goal.literals:
                if str(lit) in self.chosen_goal:
                    chosen_goal_conjunction.append(lit)

            print(f"Goal literals: {self.state.goal.literals}")
            print(f"Chosen goal: {self.chosen_goal}")
            if len(chosen_goal_conjunction) != len(self.chosen_goal):
                raise ValueError(f"Goal {self.chosen_goal} not found in the goal literals")
            self.chosen_goal = LiteralConjunction(list(chosen_goal_conjunction))

        # If you want to auto-generate and cache all possible atoms for the domain:
        # self.all_atoms = self._generate_all_atoms(self.state)
        return self.state
    
    def find_problem_num(self):
        """
        Gets a random problem that is the same domain. Very low chance of getting the same problem as the current one.
        """
        problem_nums = []
        for i in range(len(self.dataset["problems"])):
            if self.env_name.replace("PDDLEnv", "").replace("-v0","").lower()==self.dataset["problems"][i]["domain_name"]:
                problem_nums.append(i)
        
        if len(problem_nums)==0:   
            raise ValueError("No problems found for the domain")
        return random.choice(problem_nums)
      
    def find_problem_num_pred(self):
        """
        Gets a random problem that is the same domain. Very low chance of getting the same problem as the current one.
        """
        problem_nums = []
        for i in range(len(self.dataset_pred["problems"])):
            if self.env_name.replace("PDDLEnv", "").replace("-v0","").lower()==self.dataset_pred["problems"][i]["problem_name"]:
                problem_nums.append(i)
        
        if len(problem_nums)==0:   
            raise ValueError("No problems found for the domain in the prediction dataset")
        return random.choice(problem_nums)
  
        

    def step(self, state: State, action: Any) -> State:
        """
        Compute the successor state using get_successor_state from PDDLGym.

        Args:
            state: The current state.
            action: The action to perform.

        Returns:
            The next state.
        """
        next_state = get_successor_state(state, action, self.domain)
        return next_state

    def get_actions(self, state: State) -> List[Any]:
        """
        Get all possible actions from the current state.

        Args:
            state: The current state.

        Returns:
            A list of possible actions.
        """
        # Error because when using Rollouts here it has string objects in sttae smh
        actions = self.env.action_space.all_ground_literals(state)
        # Sort them in reverse lex order if desired
        return sorted(actions, key=lambda x: str(x), reverse=True)
    @staticmethod
    def string_to_12digit_number(input_string):
        """
        Converts any string into a 12-digit number.

        Args:
            input_string (str): The input string to convert.

        Returns:
            int: A 12-digit number derived from the input string.
        """
        # Use hashlib to create a SHA-256 hash of the input string
        hash_object = hashlib.sha256(input_string.encode('utf-8'))

        # Convert the hash to a hexadecimal number and then to an integer
        hash_int = int(hash_object.hexdigest(), 16)

        # Take the last 12 digits of the integer to ensure it's 12 digits long
        twelve_digit_number = hash_int % 10**12

        return twelve_digit_number
    def is_goal(self, state: State) -> bool:
        """
        Check if the state is a goal state.

        Args:
            state: The state to check.

        Returns:
            True if the state is a goal state, False otherwise.
        """
        return check_goal(state, self.chosen_goal)

    def state_to_atoms(self, state: State) -> Dict[str, Set[Literal]]:
        """
        Convert the state to a representation of atoms as a dict.

        Args:
            state: The state to convert.

        Returns:
            A dict representing the atoms in the state.
        """
        atoms = set(state.literals)
        return {"atoms": atoms}

    @staticmethod
    def state_to_grounded_predicate_strings(state: State) -> Dict[str, List[str]]:
        """
        Helper for LLM prompts that expect stringified grounded predicates.
        """
        return {"grounded_predicates": [str(lit) for lit in state.literals]}

    def _infer_fallback_object_type(self) -> Type:
        """Infer canonical fallback type for unknown SSR object names."""
        if self.state and self.state.objects:
            present_types = {
                obj.var_type
                for obj in self.state.objects
                if isinstance(getattr(obj, "var_type", None), Type)
            }
            for obj_type in present_types:
                if str(obj_type) == "default":
                    return obj_type
            if len(present_types) == 1:
                return next(iter(present_types))
        return Type("default")

    def _create_literal_from_ssr_predicate(self, predicate_name: str, raw_args: List[Any]) -> Literal:
        """
        Convert an SSR predicate (often returned with different casing) into the
        predicate objects defined in the current PDDLGym domain. Doing this in one
        place keeps the literals consistent, which is required for the action
        generator to find applicable operators during search and rollouts.

        FIX-015: Also resolves object types by looking up names in self.state.objects
        to ensure hash-consistent TypedEntity instances (a:block not a:object).
        """
        # FIX-015: Build a name→TypedEntity lookup from the ground-truth state
        gt_object_map: Dict[str, "TypedEntity"] = {}
        if self.state and self.state.objects:
            for obj in self.state.objects:
                gt_object_map[obj.name] = obj

        typed_args: List[TypedEntity] = []
        fallback_type = self._infer_fallback_object_type()
        for raw_arg in raw_args:
            if raw_arg is None:
                continue
            if isinstance(raw_arg, TypedEntity):
                # FIX-015: Even for TypedEntity, prefer GT version to fix type/hash
                if raw_arg.name in gt_object_map:
                    typed_args.append(gt_object_map[raw_arg.name])
                else:
                    typed_args.append(raw_arg)
                continue
            # Extract a name for the object.
            obj_name = getattr(raw_arg, "name", raw_arg)
            if isinstance(obj_name, TypedEntity):
                if obj_name.name in gt_object_map:
                    typed_args.append(gt_object_map[obj_name.name])
                else:
                    typed_args.append(obj_name)
                continue
            # FIX-015: Look up the GT object first to get the correct type & hash
            if obj_name in gt_object_map:
                typed_args.append(gt_object_map[obj_name])
                continue
            # Fallback: work out a reasonable type for the object.
            if hasattr(raw_arg, "var_type") and isinstance(raw_arg.var_type, Type):
                obj_type = raw_arg.var_type
            else:
                obj_type = fallback_type
            typed_args.append(TypedEntity.__new__(TypedEntity, obj_name, obj_type))

        normalized_name = predicate_name.strip().lower()
        candidate_names = [
            normalized_name,
            normalized_name.replace("_", "-"),
            normalized_name.replace("-", "_"),
            normalized_name.replace("-", ""),
            normalized_name.replace("_", ""),
        ]

        domain_predicate = None
        for candidate in candidate_names:
            if candidate in self.domain.predicates:
                domain_predicate = self.domain.predicates[candidate]
                break

        if domain_predicate is None:
            compact = normalized_name.replace("-", "").replace("_", "")
            for domain_name, predicate in self.domain.predicates.items():
                if compact == domain_name.replace("-", "").replace("_", ""):
                    domain_predicate = predicate
                    break

        if domain_predicate is None:
            domain_predicate = Predicate(normalized_name, len(typed_args))

        return Literal(domain_predicate, typed_args)

    def make_observation(self, state: State) -> Dict[str, Set[Literal]]:
        """
        Make an observation of the environment by:
          1) Using SSR to retrieve objects & predicates from an image,
          2) Converting them to a set of pddlgym.Literal for the new "observed" state,
          3) Optionally filtering out low-probability objects/predicates,
          4) Storing probability info in self.atom_probabilities.

        Args:
            state: The current true state (unused here except for naming, etc.).

        Returns:
            A dict representing the 'observed' atoms from the image + SSR.
        """
        img = self.env.unwrapped._render(state.literals)
        #make the state literals to a string
        state_literals = str(state.literals)
        #convert the state_literals to a 12 digit number
        state_hash = self.string_to_12digit_number(state_literals)
        #check if it exists
        already_seen = False
        # TODO: to save some tokens, we could check if the image already exists

        # if os.path.exists(self.image_folder_path):
        #     already_seen = True
            #maybe later save some tokens
        
        image_filename = f"observation_{state_hash}.png"
        image_path = os.path.join(self.image_folder_path, image_filename)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Convert to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Save the image
        imageio.imwrite(image_path, img)
        
        if not os.path.exists(image_path):
            # FAIL FAST — no GT/state fallback in observation path.
            error_msg = f"No image found at: {image_path}"
            logger.error(error_msg)
            raise SSRObjectRetrievalError(error_msg)

        # 1) Retrieve objects
        #    SSR returns a list (or multiple) of objects + probabilities
        #    We'll pick the top-level single answer for illustration
        #    The second return is the list of probability dicts for each object
        # Why vllm no logprobs?

        #lets save some tokens and call the objects only every 5 steps
        #do not skip if it is set to 0
        # skip if it is the current step is not zero
        
        if self.all_objects_probs_list is None:
            skip_image = False
        elif (self.num_steps_object_checking != 0 and self.step_without_checking <= self.num_steps_object_checking) or (self.num_steps_object_checking != 0 and already_seen):
            skip_image = True
        else:
            skip_image = False

        if skip_image:
            # we can skip the object retrieval if we do not want to do it every step and we either have seen the image or we are not at step 0
            # we reset for every step_without_checking if we reach
            logger.debug("SSR object retrieval skipped. Using cached objects.")
            self.step_without_checking += 1
        else:
            # Retry logic for transient VLM API errors (FIX-015b)
            max_retries = 3
            for attempt in range(max_retries):
                retrieved_objs_list, all_objects_probs_list = retrieve_objects(
                    image_path=image_path,
                    domain_name=self.env_name.replace("PDDLEnv", "").replace("-v0","").lower(),
                    dataset_path=self.dataset_path,
                    dataset=self.dataset,
                    num_examples=self.num_examples,
                    problem_id=self.problem_num,
                    number_answers=1
                )
                if retrieved_objs_list and retrieved_objs_list[0] is not None:
                    break
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2s, 4s
                    logger.warning(f"SSR object retrieval returned None (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)

            self.all_objects_probs_list = all_objects_probs_list
            self.retrieved_objs_list = retrieved_objs_list
            self.step_without_checking = 0

            # SSR returns arrays of answers; we take the first
            if not self.retrieved_objs_list or self.retrieved_objs_list[0] is None:
                # FAIL FAST — no fallbacks (after retries exhausted)
                error_msg = f"SSR object retrieval returned None for image: {image_path} (after {max_retries} attempts)"
                logger.error(error_msg)
                raise SSRObjectRetrievalError(error_msg)

        # The actual SSR object structure
        retrieved_objs = self.retrieved_objs_list[0]
        all_objs_probs = self.all_objects_probs_list[0] if self.all_objects_probs_list else []

        # Filter out objects below the threshold
        if self.object_threshold_propability > 0.0:
            all_objs_probs = get_simplified_object_probs(
                all_objs_probs, 
                object_threshold_propability=self.object_threshold_propability
            )

        
        # 2) Retrieve predicates given the objects
        # Retry logic for transient VLM API errors (FIX-015b)
        max_retries_pred = 3
        for attempt in range(max_retries_pred):
            retrieved_predicates_list, all_pred_probs_list = retrieve_predicates(
                image_path=image_path,
                domain_name=self.env_name.replace("PDDLEnv", "").replace("-v0","").lower(),
                objects=retrieved_objs,
                dataset_path=self.dataset_path,
                dataset=self.dataset,
                num_examples=self.num_examples,
                problem_id=self.problem_num,
                number_answers=1
            )
            if retrieved_predicates_list and retrieved_predicates_list[0] is not None:
                break
            if attempt < max_retries_pred - 1:
                wait_time = 2 ** (attempt + 1)
                logger.warning(f"SSR predicate retrieval returned None (attempt {attempt+1}/{max_retries_pred}), retrying in {wait_time}s...")
                import time
                time.sleep(wait_time)
        self.retrieved_objs = retrieved_objs

        if not retrieved_predicates_list or retrieved_predicates_list[0] is None:
            # FAIL FAST — no fallbacks (after retries exhausted)
            error_msg = f"SSR predicate retrieval returned None for image: {image_path} (after {max_retries_pred} attempts)"
            logger.error(error_msg)
            raise SSRPredicateRetrievalError(error_msg)

        retrieved_predicates = retrieved_predicates_list[0]
        all_pred_probs = all_pred_probs_list[0] if all_pred_probs_list else []

        # Filter out low-probability preds (optional)
        if self.object_threshold_propability > 0.0:
            all_pred_probs = get_simplified_pred_probs(
                all_pred_probs, 
                pred_threshold_propability=self.object_threshold_propability
            )
                
        # 3) Convert SSR results to pddlgym.Literal
        #    The SSR data typically has the form: 
        #    { "grounded_predicates": [ {"predicate_type": str, "x": {...}, ...}, ... ] }
        #    We'll build a set of pddlgym.structs.Literal from that.
        observed_atoms = set()
        grounded_pred_list = retrieved_predicates.grounded_predicates if hasattr(retrieved_predicates, "grounded_predicates") else []
        for gp in grounded_pred_list:
            # Example: gp = {"predicate_type": "on", "x": {"name": "block1"}, "y": {"name": "block2"} ...}
            pred_name = gp.predicate_type
            arg_sources: List[Any] = []
            for arg_label in ["x","y","z"]:
                if hasattr(gp, arg_label):
                    arg_val = gp.__getattribute__(arg_label)
                    if arg_val:
                        arg_sources.append(arg_val)
            lit = self._create_literal_from_ssr_predicate(pred_name, arg_sources)
            observed_atoms.add(lit)

        # 4) (Optional) store per-atom probabilities in self.atom_probabilities
        #    For example, you might map from "Literal(on block1 block2)" to a combined probability
        #    or store them individually. Here we do a simplified approach.
        #    We'll just store each observed atom as 1.0 for demonstration, or you can parse all_pred_probs
        self.atom_probabilities.clear()
        for lit in observed_atoms:
            self.atom_probabilities[lit] = 1.0  # or some function of all_pred_probs
        
        for pred_prob_comb in all_pred_probs:
            # Example: {'predicate': Handempty(predicate_type='Handempty'), 'prob': 0.9994970409143478}
            predic = pred_prob_comb["predicate"]
            prob = pred_prob_comb["prob"]
            
            pred_name = predic.predicate_type
            arg_sources: List[Any] = []
            for arg_label in ["x","y","z"]:
                if hasattr(predic, arg_label):
                    arg_val = predic.__getattribute__(arg_label)
                    if arg_val:
                        arg_sources.append(arg_val)
            lit = self._create_literal_from_ssr_predicate(pred_name, arg_sources)
            self.atom_probabilities[lit] = round(prob, 2)  # or some function of all_pred_probs
        
        # Return the observation
        return {"atoms": observed_atoms}

    def atoms_to_state(self, atoms: Dict[str, Set[Literal]]) -> State:
        """
        Convert a dict of atoms to a State object.

        Args:
            atoms: The atoms to convert. Expected format:
                   {"atoms": set_of_literals}

        Returns:
            The State represented by the atoms.

        Raises:
            ValueError: If the required keys are not present in the atoms dict.
            AttributeError: If the current state is not initialized to extract the goal.
        """
        if "atoms" not in atoms:
            raise ValueError("The atoms dictionary must contain the key 'atoms'.")

        literals: Set[Literal] = frozenset(atoms["atoms"])

        # Use the objects from the initial state to maintain compatibility with PDDLGym
        # This is necessary because PDDLGym expects all states to have the same objects
        # ATTENTION!! HERE IN THE STATES THEY WILL USE THE GT OBJECTS
        objects = self.state.objects if self.state else None
        if objects is None:
            # Fall back to extracting objects from literals if we don't have the original objects
            objects_set = set()
            for lit in literals:
                # If the literal is something like on(block1, block2)
                # the .variables are the arguments. 
                # In pddlgym, that's lit.variables
                objects_set.update(lit.variables)
            objects = frozenset(objects_set)

        # In a typical PDDLGym State, the 'goal' field is carried from the environment
        # We'll assume the self.state has a valid .goal
        if self.state and hasattr(self.state, 'goal'):
            goal = self.state.goal
        else:
            raise AttributeError("Current state is not initialized or does not have a goal.")

        # Create and return the new State
        new_state = State(literals, objects, goal)
        return new_state

    def get_state_probability(self, state: State) -> Dict[Literal, float]:
        """
        Get the per-atom probabilities for the current sampled (observed) state.

        Args:
            state: The observed state (not always used directly).

        Returns:
            A dictionary mapping atoms to their probabilities.
        """
        return self.atom_probabilities

    def get_atom_probability(self, atom: Literal) -> float:
        """
        Get the probability of the atom being observed as true.

        Args:
            atom: The atom (Literal object).

        Returns:
            The probability that the atom is observed as true.
        """
        return self.atom_probabilities.get(atom, 0.0)

    def predict_next_state(self, observed_state: State, action: Any) -> State:
        """
        Predict the next state given the observed state and action.
        
        This uses the SSR 'predict_next_state' method, 
        which is an LLM-based approach to generating new symbolic facts.

        Args:
            observed_state: The observed (possibly noisy) state as a pddlgym State.
            action: The action taken (string or pddlgym Literal).

        Returns:
            The predicted next state as a pddlgym.State.
        """
        # 1) Convert observed_state to dict format for SSR
        observed_atoms_dict = self.state_to_atoms(observed_state)
        # SSR expects something like: {"grounded_predicates":[ ... ]}
        # We'll convert each pddlgym Literal to a generic dict
        # and pass it to SSR's predict_next_state.
        # Build a simpler structure for SSR
        grounded_predicates_for_ssr = []
        for lit in observed_atoms_dict["atoms"]:
            pred_name = lit.predicate.name
            args = list(lit.variables)
            # SSR typically expects a dictionary with "predicate_type" and x, y, z
            # We'll accommodate up to 3 arguments for demonstration
            entry = {
                "predicate_type": pred_name,
            }
            # Map arguments to x,y,z if present
            if len(args) > 0:
                entry["x"] = {"name": args[0]}
            if len(args) > 1:
                entry["y"] = {"name": args[1]}
            if len(args) > 2:
                entry["z"] = {"name": args[2]}
            grounded_predicates_for_ssr.append(entry)

        # Now we have something like:
        # current_state_for_ssr = {"grounded_predicates": grounded_predicates_for_ssr}
        current_state_for_ssr = {"grounded_predicates": grounded_predicates_for_ssr}
        # 2) Call SSR's predict_next_state
        # SSR returns something like ([instance], [list_of_pred_probs]) if number_answers=1
        predicted_atoms_list, predicted_atoms_probs_list = ssr_predict_next_state(
            action=str(action),  # or action.predicate.name if needed
            current_state=current_state_for_ssr,
            objects=self.retrieved_objs,
            domain_name=self.env_name.replace("PDDLEnv", "").replace("-v0","").lower(),
            dataset_path=self.next_state_dataset_path,
            dataset=self.dataset_pred,  # if you have a dataset loaded
            num_examples=self.num_examples,
            number_answers=1,
        )

        if not predicted_atoms_list or predicted_atoms_list[0] is None:
            error_msg = "SSR next-state prediction returned None"
            logger.error(error_msg)
            raise SSRPredicateRetrievalError(error_msg)

        predicted_atoms_struct = predicted_atoms_list[0]
        # predicted_atoms_probs = predicted_atoms_probs_list[0]  # if you want to store them

        # 3) Convert SSR’s predicted_atoms_struct to a pddlgym.State
        # predicted_atoms_struct might have the form:  {"grounded_predicates":[ {...}, {...} ]}
        predicted_atoms = set()
        if hasattr(predicted_atoms_struct, "grounded_predicates"):
            for gp in predicted_atoms_struct.grounded_predicates:
                pred_name = gp.predicate_type
                arg_sources: List[Any] = []
                for arg_label in ["x","y","z"]:
                    if hasattr(gp, arg_label):
                        arg_val = gp.__getattribute__(arg_label)
                        if arg_val:
                            arg_sources.append(arg_val)
                lit = self._create_literal_from_ssr_predicate(pred_name, arg_sources)
                predicted_atoms.add(lit)

        # 4) Construct a new State
        # Reuse the same objects in the environment or from the predicted arguments
        objects_set = set()
        for lit in predicted_atoms:
            objects_set.update(lit.variables)
        objects_frozen = frozenset(objects_set)
        # Keep the same goal as the original environment (if desired)
        goal = self.state.goal if (self.state and hasattr(self.state, "goal")) else None

        predicted_next_state = State(frozenset(predicted_atoms), objects_frozen, goal)
        return predicted_next_state

    def choose_best_action(self, state: State, actions: List[Any]) -> Optional[Any]:
        """
        Use the LLM to select the most promising action from the provided list.
        Returns the original action literal when possible.
        """
        if not actions:
            return None

        action_strings = [str(action) for action in actions]
        action_lookup = {str(action): action for action in actions}
        current_state_struct = self.state_to_grounded_predicate_strings(state)

        result = ssr_select_best_action(
            actions=action_strings,
            current_state=current_state_struct,
            goal=self.chosen_goal,
            domain_name=self.env_name.replace("PDDLEnv", "").replace("-v0", "").lower(),
        )

        if not result:
            return None

        best_action_str = result.get("best_action")
        return action_lookup.get(best_action_str)

    def score_state_with_heuristic(self, state: State) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM to score the current state on a 1-10 heuristic scale.
        """
        current_state_struct = self.state_to_grounded_predicate_strings(state)
        result = ssr_score_state_heuristic(
            current_state=current_state_struct,
            goal=self.chosen_goal,
            domain_name=self.env_name.replace("PDDLEnv", "").replace("-v0", "").lower(),
        )
        return result

    def _generate_all_atoms(self, state: State) -> Set[Literal]:
        """
        Generate all possible ground atoms for the current domain and state objects.
        Useful if you want a reference set of all possible atoms.

        Args:
            state: A State from which to pull objects.

        Returns:
            A set of all possible ground atoms (pddlgym.structs.Literal) for the domain.
        """
        all_atoms = set()
        # For each predicate in the domain:
        for pred in self.domain.predicates:
            # pred is a pddlgym.structs.Predicate
            # Find all combinations of objects that match the arity
            arity = pred.arity
            # The environment has state objects; we can do combinations if needed:
            for combo in itertools.product(state.objects, repeat=arity):
                # A quick check if the domain might have typed objects
                # or constraints, which you might need to check.
                # We'll skip type-checking for brevity.
                # Build a literal
                lit = Literal(pred, combo, is_negative=False)
                all_atoms.add(lit)
        return all_atoms


if __name__ == "__main__":
    # Illustrative demo of the VLM-grounded simulator on a bundled paper domain.
    # For the reproducible VLM-grounded planning entrypoint use
    #   python -m experiments.run_search_comparison --env PDDLEnvHanoi-v0
    domain_name = "blocks_operator_actions"  # Blocksworld (paper PDDLGym domain)
    action = "move_block_a_to_block_b"       # Example action
    dataset_path = "./experiments/data/datasets/pddlGYM_dataset_grounding.json"
    num_examples = 1
    next_state_dataset_path = "./experiments/data/datasets/pddlGYM_dataset_grounding.json"
    problem_index = 2
    chosen_goal = ["on(yellow:block,purple:block)"]
    env_name = f"PDDLEnv{domain_name.capitalize()}-v0"
    image_folder_path = "./datasets/images"
    
    
    simulator = PDDLGymSimulatorOpenai(
        env_name=env_name,
        dataset_path=dataset_path,
        image_folder_path=image_folder_path,
        next_state_dataset_path=next_state_dataset_path,
        num_examples=num_examples,
        problem_index=problem_index,
        object_threshold_propability=0.2,
        use_original_objects=False,
        chosen_goal=chosen_goal
    )
    # simulator = PDDLGymSimulator(
    #     env_name=env_name,
    #     problem_index=problem_index,
    #     chosen_goal=chosen_goal
    # )
    ### TODO: WHY IS IT USING ./experiments/ssr_eval/pddlGymExperiment/data/hanoi/observations/problem4.jpg for an blocks image!!!
    # Example usage:
    init_state = simulator.reset()
    print("Initialized State")
    observation = simulator.make_observation(init_state)
    print("Made Observation")
    
    obs_state = simulator.atoms_to_state(observation)
    print("Observed State")
    next_state_pred = simulator.predict_next_state(obs_state, action)
    print("Observed Atoms:", observation["atoms"])
    print("Predicted Next State Atoms:", next_state_pred.literals)
    	
