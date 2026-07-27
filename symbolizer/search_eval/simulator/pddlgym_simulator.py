import random
import itertools
from typing import Any, List, Dict, Set
from pddlgym import make
from pddlgym.inference import check_goal
from pddlgym.core import get_successor_state
from pddlgym.structs import State, Literal, Predicate


class PDDLGymSimulator:
    """
    Simulator implementation using PDDL Gym with probabilistic noise.
    This simulator introduces false positive and false negative noise
    when making observations of the environment's state.
    """

    def __init__(
        self,
        env_name: str,
        problem_index: int = 0,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        total_atoms: int = None,
        prediction_false_positive_rate: float = 0.0,
        prediction_false_negative_rate: float = 0.0, 
        chosen_goal = None
    ):
        """
        Initialize the PDDLGym simulator with probabilistic noise.

        Args:
            env_name: The name of the PDDL Gym environment.
            problem_index: The index of the problem to load.
            false_positive_rate: Average probability that a false atom becomes a false positive.
            false_negative_rate: Average probability that a true atom becomes a false negative.
            total_atoms: Total number of possible atoms in the domain.
                         If not provided, it will be estimated.
        """
        self.env_name = env_name
        self.problem_index = problem_index
        self.env = make(env_name)
        self.env.fix_problem_index(problem_index)
        self.state = None
        self.domain = self.env.domain
        self.chosen_goal = chosen_goal

        # Noise parameters
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate

        #Noise parameters for prediction
        self.prediction_false_positive_rate = prediction_false_positive_rate
        self.prediction_false_negative_rate = prediction_false_negative_rate
        
        # Total number of possible atoms
        self.total_atoms = total_atoms

        # Cache for all possible atoms
        self.all_atoms: Set[Literal] = set()

        # Dictionary to store per-atom probabilities
        self.atom_probabilities: Dict[Literal, float] = {}
        
        # Dictionary to store per-atom probabilities for prediction
        self.atom_probabilities_prediction: Dict[Literal, float] = {}

    def _generate_all_atoms(self) -> Set[Literal]:
        """
        Generate all possible ground atoms based on the domain's predicates and objects.

        Returns:
            A set of Literal objects representing all possible atoms.
        """
        all_atoms = set()
        objects = list(self.state.objects)

        for predicate in self.domain.predicates.values():
            arity = predicate.arity
            if arity == 0:
                # Propositional predicate
                atom = Literal(predicate, [])
                all_atoms.add(atom)
            else:
                # Generate all possible combinations of objects for the predicate's arity
                for args in itertools.product(objects, repeat=arity):
                    atom = Literal(predicate, args)
                    all_atoms.add(atom)
        return all_atoms

    def _estimate_total_atoms(self) -> int:
        """
        Estimate the total number of possible ground atoms in the domain.

        Returns:
            An integer representing the total number of possible atoms.
        """
        total = 0
        objects = list(self.state.objects)
        num_objects = len(objects)
        for predicate in self.domain.predicates.values():
            arity = predicate.arity
            if arity == 0:
                total += 1
            else:
                total += num_objects ** arity
        return total

    def reset(self) -> State:
        """
        Reset the simulator to the initial state.

        Returns:
            The initial state.
        """
        self.state, _ = self.env.reset()
        # Total number of possible atoms
        if self.total_atoms is None:
            self.total_atoms = self._estimate_total_atoms()
        if not self.chosen_goal:
            self.chosen_goal = self.state.goal
        if isinstance(self.chosen_goal,str):
            for lit in self.state.goal.literals:
                if str(lit) == self.chosen_goal:
                    self.chosen_goal = lit
                    break
            if isinstance(self.chosen_goal, str):
                print(self.state.goal.literals)
                raise ValueError(f"Goal {self.chosen_goal} not found in the goal literals")
        # Generate and cache all possible atoms
        self.all_atoms = self._generate_all_atoms()
        return self.state

    def step(self, state: State, action: Any) -> State:
        """
        Compute the successor state using get_successor_state.

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
        actions = self.env.action_space.all_ground_literals(state)
        return sorted(actions, key=lambda x: str(x), reverse=True)

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

    def make_observation(self, state: State) -> Dict[str, Set[Literal]]:
        """
        Make an observation of the state by applying noise.

        Args:
            state: The state to observe.

        Returns:
            A dict representing the noisy observation.
        """
        sampled_state = self.sample_state(state)
        return self.state_to_atoms(sampled_state)

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

        # # Extract objects from literals
        # objects_set = set()
        # for lit in literals:
        #     if isinstance(lit, Predicate):
        #         continue
        #     objects_set.update(lit.variables)
        # objects = frozenset(objects_set)

        # Extract the goal from the current state if available
        if self.state and hasattr(self.state, 'goal'):
            goal = self.chosen_goal
        else:
            raise AttributeError("Current state is not initialized or does not have a goal.")

        # Create and return the new State
        new_state = State(literals, self.state.objects, goal)
        return new_state

    def get_state_probability(self, state: State) -> Dict[Literal, float]:
        """
        Get the per-atom probabilities for the current sampled state.

        Args:
            state: The observed state.

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

    def sample_state(self, state: State) -> State:
        atoms_present = set(state.literals)
        atoms_not_present = self.all_atoms - atoms_present

        P_false_negative = self.false_negative_rate
        P_false_positive = self.false_positive_rate

        observed_atoms = set()
        self.atom_probabilities = {}

        # True atoms (might become false negatives)
        for atom in atoms_present:
            #add normal distribution noise 
            prob = min(1,max(0,1-P_false_negative*random.normalvariate(1,0.3)))
            self.atom_probabilities[atom] = prob
            if random.random() < prob:
                observed_atoms.add(atom)

        len_false_positives = len(atoms_present) * self.false_positive_rate
        #randomly choose len_false_positives atoms to be false positives
        for atom in random.sample(list(atoms_not_present), int(len_false_positives)): 
            prob = min(1,max(0,P_false_positive*random.normalvariate(1,0.3)))
            self.atom_probabilities[atom] = prob
            if random.random() < prob and prob > 0.01:
                observed_atoms.add(atom)

        return self.atoms_to_state({"atoms": observed_atoms})

    def sample_state_prediction(self, state: State) -> State:
        atoms_present = set(state.literals)
        atoms_not_present = self.all_atoms - atoms_present

        P_true_positive = 1.0 - self.prediction_false_negative_rate
        P_false_positive = self.prediction_false_positive_rate

        observed_atoms = set()
        self.atom_probabilities_prediction = {}

        # True atoms (might become false negatives)
        for atom in atoms_present:
            prob = P_true_positive
            self.atom_probabilities_prediction[atom] = prob
            if random.random() < prob:
                observed_atoms.add(atom)

        # False atoms (might become false positives)
        for atom in atoms_not_present:
            prob = P_false_positive
            self.atom_probabilities_prediction[atom] = prob
            if random.random() < prob:
                observed_atoms.add(atom)

        return self.atoms_to_state({"atoms": observed_atoms})

    def predict_next_state(self, observed_state: State, action: Any) -> State:
        """
        Predict the next state given the observed state and action.

        Args:
            observed_state: The observed (noisy) state.
            action: The action taken.

        Returns:
            The predicted next state.
        """
        # Check if the action is applicable in the observed state
        applicable_actions = self.get_actions(observed_state)
        if action in applicable_actions:
            predicted_next_state = get_successor_state(
                observed_state,
                action,
                self.domain,
                require_unique_assignment=False
            )    
        else:
            # If action is not applicable, return the observed state unchanged
            predicted_next_state = observed_state
        
        # Apply noise to the predicted state
        predicted_next_state = self.sample_state_prediction(predicted_next_state)
        
        return predicted_next_state


