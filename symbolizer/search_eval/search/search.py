# Search/search.py

from abc import ABC, abstractmethod
from typing import Any, List


class SearchAlgorithm(ABC):
    """
    Abstract Search Algorithm class.
    """

    @abstractmethod
    def search(self, simulator: Any, initial_state: Any) -> List[Any]:
        """
        Perform the search to find a path to a goal state.

        Args:
            simulator: The simulator to interact with.
            initial_state: The initial state to start from.

        Returns:
            A list of actions representing the path to a goal state.
        """
        pass
