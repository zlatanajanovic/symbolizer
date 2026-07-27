from abc import ABC, abstractmethod
from typing import Any, List, Dict, Set, Tuple, Optional

class Simulator(ABC):
    """
    Abstract base class for PDDL Simulators.
    """

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        prediction_false_positive_rate: float = 0.0,
        prediction_false_negative_rate: float = 0.0
    ):
        """
        Initialize the PDDL simulator with optional false positive and negative rates for observations and predictions.

        :param domain_file: Path to the PDDL domain file.
        :param problem_file: Path to the PDDL problem file.
        :param false_positive_rate: False positive rate for observations (0.0 to 1.0).
        :param false_negative_rate: False negative rate for observations (0.0 to 1.0).
        :param prediction_false_positive_rate: False positive rate for predictions (0.0 to 1.0).
        :param prediction_false_negative_rate: False negative rate for predictions (0.0 to 1.0).
        """
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.prediction_false_positive_rate = prediction_false_positive_rate
        self.prediction_false_negative_rate = prediction_false_negative_rate


    @abstractmethod
    def reset(self) -> Set[Tuple[str, Tuple[int, ...]]]:
        """
        Reset the simulator to the initial state.

        :return: A copy of the initial state.
        """
        pass

    @abstractmethod
    def is_goal(self, state: Set[Tuple[str, Tuple[int, ...]]]) -> bool:
        """
        Check if the given state is a goal state.

        :param state: The current state represented as a set of (predicate_name, args).
        :return: True if the state satisfies the goal, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, state: Set[Tuple[str, Tuple[int, ...]]], action: Tuple[str, Tuple[int, ...]]) -> Set[Tuple[str, Tuple[int, ...]]]:
        """
        Apply an action to the state and return the next state.

        :param state: The current state.
        :param action: The action to apply.
        :return: The next state after applying the action.
        """
        pass

    @abstractmethod
    def get_successor_state(self, state: Set[Tuple[str, Tuple[int, ...]]], action: Tuple[str, Tuple[int, ...]]) -> Set[Tuple[str, Tuple[int, ...]]]:
        """
        Compute the successor state given a state and an action.

        :param state: The current state represented as a set of (predicate_name, args).
        :param action: The action to apply, as a tuple (action_name, args).
        :return: The successor state as a set of (predicate_name, args).
        """
        pass

    @abstractmethod
    def get_actions(self, state: Set[Tuple[str, Tuple[int, ...]]]) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Get the list of possible actions from a given state.

        :param state: The current state represented as a set of (predicate_name, args).
        :return: A list of applicable actions as tuples (action_name, args).
        """
        pass

    @abstractmethod
    def make_observation(self, state: Set[Tuple[str, Tuple[int, ...]]]) -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
        """
        Make an observation of the given state with noise.

        :param state: The current state represented as a set of (predicate_name, args).
        :return: A dictionary containing the observed atoms.
        """
        pass

    @abstractmethod
    def predict_next_state(self, observed_state: Set[Tuple[str, Tuple[int, ...]]], action: Tuple[str, Tuple[int, ...]]) -> Set[Tuple[str, Tuple[int, ...]]]:
        """
        Predict the next state given an observed state and action, incorporating prediction noise.

        :param observed_state: The observed state represented as a set of (predicate_name, args).
        :param action: The action to apply, as a tuple (action_name, args).
        :return: The predicted next state as a set of (predicate_name, args).
        """
        pass

    @abstractmethod
    def _introduce_noise(
        self,
        true_atoms: Set[Tuple[str, Tuple[str, ...]]],
        false_positive_rate: float,
        false_negative_rate: float
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        """
        Introduce false positives and false negatives into the true atoms.

        :param true_atoms: The set of true atoms (using object names).
        :param false_positive_rate: The false positive rate (probability of adding a false atom).
        :param false_negative_rate: The false negative rate (probability of removing a true atom).
        :return: A list of atoms after introducing noise.
        """
        pass

    @abstractmethod
    def _get_all_possible_atoms_names(self) -> Set[Tuple[str, Tuple[str, ...]]]:
        """
        Generate all possible grounded atoms based on predicates and objects, using object names.

        :return: A set of all possible atoms (using object names).
        """
        pass

    @abstractmethod
    def state_to_atoms(self, state: Set[Tuple[str, Tuple[int, ...]]]) -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
        """
        Convert the state to a representation of atoms with object names.

        :param state: The state represented as a set of (predicate_name, args).
        :return: A dictionary containing the atoms with object names.
        """
        pass

    @abstractmethod
    def atoms_to_state(self, atoms: Dict[str, List[Tuple[str, Tuple[str, ...]]]]) -> Set[Tuple[str, Tuple[int, ...]]]:
        """
        Convert a dict of atoms with object names to a state represented with object indices.

        :param atoms: A dictionary containing the atoms with object names.
        :return: A set of atoms representing the state with object indices.
        """
        pass
