import math
import random
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional, FrozenSet

# Optional: for Beta distribution numeric integration
import numpy as np
from scipy.stats import beta
from scipy.integrate import quad,fixed_quad
from scipy.optimize import brentq

class Novelty(ABC):
    @abstractmethod
    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        """
        Compute a novelty score based on feature tuples.
        """
        pass

    @abstractmethod
    def is_novel(self, novelty_value: float) -> bool:
        """
        Decide if a given novelty value means the state is novel.
        """
        pass


class NoveltyNaiveChooseAny1(Novelty):
    """
    Implements the Boolean novelty from eq:npowbs_any_1.
    floor(1 + log(sensor_model(state, ft)) - log(closed_list(ft)) - log(threshold_novel))
    If >= 1 => TRUE (novel), else FALSE
    Combines over all feature tuples by checking if ANY tuple triggers novelty.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.threshold_novel = threshold_novel

    def compute_novelty(
        self, 
        observation: Any, 
        feature_tuples: List[Tuple[Any, ...]], 
        atom_probs: Dict[Any, float], 
        closed_list
    ) -> float:
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            p = closed_list.get_probability(ft_key)  # Probability stored in closed_list
            # Compute joint probability from atom_probs
            ft_prob = 1.0
            for atom in ft:
                ft_prob *= atom_probs.get(atom, 0.0)

            # Probability that feature tuple was NOT previously in the closed list
            # c = 1 - P(ft ∈ closed_list) as per paper eq:npowbs_any_1
            c = 1.0 - p

            # Avoid log(0) by adding a small epsilon
            epsilon = 1e-12
            # Formula: floor(1 + log(p_sensor) - log(c) - log(threshold))
            # where c = 1 - closed_list_prob (probability tuple is NEW)
            val = 1.0 + math.log(ft_prob + epsilon) - math.log(c + epsilon) - math.log(self.threshold_novel + epsilon)

            # If val >= 1.0 => we treat that as novelty = 1
            if val >= 1.0:
                return 1.0
        return 0.0

    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value >= 1.0


class NoveltyNaiveChooseAny2(Novelty):
    """
    Implements eq:npowbs_any_2.
    floor(1 - product(1 - sensor_model(ft)*(1 - closed_list(ft))) + (1 - threshold_novel)).
    If >= 1 => True.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.threshold_novel = threshold_novel

    def compute_novelty(
        self, 
        observation: Any, 
        feature_tuples: List[Tuple[Any, ...]], 
        atom_probs: Dict[Any, float], 
        closed_list
    ) -> float:
        product_val = 1.0
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            ft_prob = 1.0
            for atom in ft:
                ft_prob *= atom_probs.get(atom, 0.0)
            c = 1.0 - closed_list.get_probability(ft_key)
            term = 1.0 - (ft_prob * c)
            product_val *= term

        val = 1.0 - product_val + (1.0 - self.threshold_novel)
        return math.floor(val)

    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value >= 1.0


class NoveltyNaiveChooseMax1(Novelty):
    """
    Implements eq:npowbs_max_1.
    max(log(sensor_model(ft)) - log(1-closed_list(ft)) - log(threshold_novel)) among all feature tuples.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.threshold_novel = threshold_novel

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        max_val = -float('inf')
        epsilon = 1e-12
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            ft_prob = 1.0
            for atom in ft:
                ft_prob *= atom_probs.get(atom, 0.0)
            c = 1.0 - closed_list.get_probability(ft_key)

            if ft_prob > 0 and c > 0:
                val = math.log(ft_prob + epsilon) - math.log(c + epsilon) - math.log(self.threshold_novel + epsilon)
                if val > max_val:
                    max_val = val

        if max_val == -float('inf'):
            # No valid feature tuples found => extremely negative
            max_val = -1000.0
        return max_val

    def is_novel(self, novelty_value: float) -> bool:
        # If novelty_value > 0 => novel
        return novelty_value >= 0.0


class NoveltyNaiveChooseMax2(Novelty):
    """
    Implements eq:npowbs_max_2.
    1 - product(1 - sensor_model(ft)*(1 - closed_list(ft))) + (1-threshold_novel).
    If > 1 => novel.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.threshold_novel = threshold_novel

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        product_val = 1.0
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            ft_prob = 1.0
            for atom in ft:
                ft_prob *= atom_probs.get(atom, 0.0)
            c = 1.0 - closed_list.get_probability(ft_key)
            product_val *= (1.0 - ft_prob * c)

        val = 1.0 - product_val + (1.0 - self.threshold_novel)
        return val

    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value >= 1.0


class NoveltyNaiveChooseMax3(Novelty):
    """
    Implements eq:npowbs_max_3.
    Sum of [sensor_model(ft)*(1 - closed_list(ft))] over all ft. If sum > threshold_novel => novel.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.threshold_novel = threshold_novel

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        total = 0.0
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            ft_prob = 1.0
            for atom in ft:
                #print("atom: ", atom)
                ft_prob *= atom_probs.get(atom, 0.0)
            #print("ft_key: ", ft_key)
            #print("ft_prob ", ft_prob)
            c = 1.0 - closed_list.get_probability(ft_key)
            #print("c: ", c)
            total += ft_prob * c
            #print("total: ", total)
        return total

    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value > self.threshold_novel


class NoveltyMonteCarlo(Novelty):
    """
    Monte Carlo novelty from eq:npowbs_mcs.
    We do multiple simulations, each time sampling whether each feature tuple is present 
    based on atom_probs. Then apply a base boolean novelty function to the sampled set.
    Average the results over multiple simulations.
    """

    def __init__(
        self,
        base_boolean_novelty: Novelty,
        number_simulations: int = 100,
        seed: Optional[int] = None
    ):
        self.base_boolean_novelty = base_boolean_novelty
        self.number_simulations = number_simulations
        if seed is not None:
            random.seed(seed)

    def compute_novelty(
        self, 
        observation: Any, 
        feature_tuples: List[Tuple[Any, ...]], 
        atom_probs: Dict[Any, float], 
        closed_list
    ) -> float:
        count_novel = 0
        for _ in range(self.number_simulations):
            sampled_feature_tuples = []
            for ft in feature_tuples:
                present = True
                for atom in ft:
                    # sample whether the atom is present
                    if random.random() >= atom_probs.get(atom, 0.0):
                        present = False
                        break
                if present:
                    sampled_feature_tuples.append(ft)
            # Compute novelty with the base boolean novelty function
            val = self.base_boolean_novelty.compute_novelty(
                observation=observation,
                feature_tuples=sampled_feature_tuples,
                atom_probs=atom_probs,
                closed_list=closed_list
            )
            if self.base_boolean_novelty.is_novel(val):
                count_novel += 1
        return count_novel / self.number_simulations

    def is_novel(self, novelty_value: float) -> bool:
        # If average probability > 0.5 => considered novel
        return novelty_value > 0.5



class NoveltyIWRollout(Novelty):
    """
    Runs IW-style rollouts driven by the simulator's predicted transitions. A rollout
    yields a novelty bonus if it encounters any feature tuple absent from the closed
    list, and a goal bonus if it reaches a goal state.
    """

    def __init__(
        self,
        simulator,
        width: int = 2,
        rollout_depth: int = 5,
        number_rollouts: int = 10,
        novelty_bonus: float = 0.9,
        goal_bonus: float = 5.0,
        closed_list_threshold: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.simulator = simulator
        self.width = max(1, int(width))
        self.rollout_depth = rollout_depth
        self.number_rollouts = number_rollouts
        self.novelty_bonus = novelty_bonus
        self.goal_bonus = goal_bonus
        self.closed_list_threshold = closed_list_threshold
        if seed is not None:
            random.seed(seed)

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        if not self.simulator or not hasattr(self.simulator, "predict_next_state"):
            return 0.0

        if self.simulator.is_goal(observation):
            return self.goal_bonus

        if self.number_rollouts <= 0:
            return 0.0

        total_reward = 0.0
        for _ in range(self.number_rollouts):
            total_reward += self._simulate_rollout(
                observation,
                feature_tuples,
                closed_list,
            )
        return total_reward / float(self.number_rollouts)

    def _simulate_rollout(
        self,
        start_state: Any,
        initial_feature_tuples: List[Tuple[Any, ...]],
        closed_list,
    ) -> float:
        current_state = start_state
        rollout_seen: set[FrozenSet[Any]] = set()
        novelty_found = self._record_feature_tuples(
            initial_feature_tuples,
            rollout_seen,
            closed_list,
        )

        for _ in range(self.rollout_depth):
            if self.simulator.is_goal(current_state):
                return self.goal_bonus

            actions = self.simulator.get_actions(current_state)
            if not actions:
                break

            action = random.choice(actions)
            next_state = self.simulator.predict_next_state(current_state, action)
            current_state = next_state

            literals = getattr(current_state, "literals", None)
            if literals is None:
                break

            new_feature_tuples = self._extract_feature_tuples(list(literals))
            if self._record_feature_tuples(new_feature_tuples, rollout_seen, closed_list):
                novelty_found = True

        if novelty_found:
            print(f"[DEBUG] Novelty found in rollout: {rollout_seen}")
            return self.novelty_bonus
        return 0.0

    def _record_feature_tuples(
        self,
        feature_tuples: List[Tuple[Any, ...]],
        rollout_seen: set,
        closed_list,
    ) -> bool:
        novel = False
        if not feature_tuples:
            return False

        for ft in feature_tuples:
            ft_key = frozenset(ft)
            if ft_key in rollout_seen:
                continue
            rollout_seen.add(ft_key)

            probability = 0.0
            if closed_list is not None and hasattr(closed_list, "get_probability"):
                probability = closed_list.get_probability(ft_key)
            if probability <= self.closed_list_threshold:
                novel = True
        return novel

    def _extract_feature_tuples(self, atoms: List[Any]) -> List[Tuple[Any, ...]]:
        atoms_sorted = tuple(sorted(atoms, key=str))
        tuples: List[Tuple[Any, ...]] = []
        for w in range(1, self.width + 1):
            tuples.extend(combinations(atoms_sorted, w))
        return tuples

    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value > 0.0


class ProbabilisticNoveltyMax0(Novelty):
    """
    A simple novelty measure that takes the maximum 'novelty probability'
    across all feature tuples:
        novelty_prob = max over ft of { (1 - closed_list(ft)) * joint_probability(ft) }
    If > threshold => novel.
    """

    def __init__(self, threshold_novel: float = 0.1):
        self.novelty_threshold = threshold_novel

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        max_novelty = 0.0
        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            ft_prob = 1.0
            for atom in ft:
                ft_prob *= atom_probs.get(atom, 0.0)
            # Probability that it wasn't in the closed list
            prob_not_in = 1.0 - closed_list.get_probability(ft_key)
            novelty = prob_not_in * ft_prob
            if novelty > max_novelty:
                max_novelty = novelty
        return max_novelty
    
    def is_novel(self, novelty_value: float) -> bool:
        return novelty_value > self.novelty_threshold


###############################################################################
#                           NEW CLASSES BELOW                                 #
###############################################################################



class NoveltyBetaDistribution(Novelty):
    def __init__(
        self,
        threshold_novel: float = 0.1,
        false_positive_rate: float = 0.1,
        false_negative_rate: float = 0.1,
        variance_scale: float = 1.0  # New parameter to scale variance
    ):
        self.threshold_novel = threshold_novel
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.variance_scale = variance_scale  # Scaling factor for variance

    def beta_params_from_mean(self, mu: float, fpr: float, fnr: float):
        # Adjust variance calculation with scaling factor
        variance = mu * (1 - mu) * ((fpr + fnr) / 2.0) * self.variance_scale

        # Add a minimum variance floor to prevent too small variance
        min_variance = 1e-2  # Adjust as needed
        if variance < min_variance:
            variance = min_variance

        ab = (mu * (1.0 - mu)) / variance - 1.0
        ab = max(ab, 2.0)  # Ensure ab is at least 2 to maintain reasonable spread

        alpha = mu * ab
        beta_ = (1.0 - mu) * ab

        # Clamp alpha and beta to prevent them from being too large
        max_alpha_beta = 1e3  # Adjust based on your domain
        alpha = min(max(alpha, 1e-4), max_alpha_beta)
        beta_ = min(max(beta_, 1e-4), max_alpha_beta)

        return alpha, beta_

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        max_beta_novelty = 0.0

        for ft in feature_tuples:
            ft_key = frozenset(sorted(ft))
            # Compute joint probability for the feature tuple
            ft_prob = 1.0
            for atom in ft:
                atom_prob = atom_probs.get(atom, 0.0)
                ft_prob *= atom_prob

            if ft_prob <= 0.0:
                continue

            p_closed = closed_list.get_probability(ft_key)

            # Handle edge cases explicitly
            if p_closed == 0.0:
                integral_val = 1.0
            elif p_closed == 1.0:
                integral_val = 0.0
            else:
                # Compute Beta params for obs and closed
                alpha_obs, beta_obs = self.beta_params_from_mean(
                    ft_prob, self.false_positive_rate, self.false_negative_rate
                )
                alpha_cl, beta_cl = self.beta_params_from_mean(
                    p_closed, self.false_positive_rate, self.false_negative_rate
                )

                # Compute P(X_obs > X_closed)
                # Define integrand for novelty: BetaPDF_obs(x) * (1 - BetaCDF_cl(x))
                def integrand(x):
                    return beta.pdf(x, alpha_obs, beta_obs) * (1.0 - beta.cdf(x, alpha_cl, beta_cl))

                integral_val, error = quad(integrand, 0, 1, limit=100)

            if integral_val > max_beta_novelty:
                max_beta_novelty = integral_val

        return max_beta_novelty

    def is_novel(self, novelty_value: float) -> bool:
        is_novel = novelty_value > self.threshold_novel
        return is_novel



class NoveltyMCTSInRollout(Novelty):
    """
    A full MCTS-inspired rollout novelty class. 
    Each call to compute_novelty(...) triggers multiple MCTS-like rollouts from the current state.
    
    The final novelty is the average "rollout reward" over all rollouts, where:
      rollout_reward = (1.0 if a goal state is reached) + novelty_bonus_for_non-goal
    We interpret novelty_bonus as some function of state if it's not a goal.
    
    If used as a direct novelty measure, the node's "novelty" is how likely it is to lead 
    to a goal or to a novel region within a small search horizon.
    """

    def __init__(
        self,
        simulator,
        base_novelty, 
        rollout_depth: int = 5,
        number_rollouts: int = 10,
        novelty_bonus: float = 0.9,
        threshold_novel: float = 0.6,
        seed: Optional[int] = None
    ):
        """
        Args:
            simulator: An object with .predict_next_state(...), .get_actions(...), .is_goal(...).
            base_novelty: Another Novelty instance used to compute a 'base novelty measure' at each state 
                          (e.g., NoveltyNaiveChooseMax3).
            rollout_depth: Maximum depth of each rollout.
            number_rollouts: Number of rollout simulations to perform.
            novelty_bonus: How much reward to add to each rollout if the state is considered novel.
            threshold_novel: A threshold to consider a state 'novel enough' to add novelty_bonus.
            seed: For reproducible random rollouts.
        """
        self.simulator = simulator
        self.base_novelty = base_novelty
        self.rollout_depth = rollout_depth
        self.number_rollouts = number_rollouts
        self.novelty_bonus = novelty_bonus
        self.threshold_novel = threshold_novel

        if seed is not None:
            random.seed(seed)

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        """
        Perform MCTS-like rollouts from 'state' up to 'rollout_depth', 
        repeating 'number_rollouts' times.
        
        In each rollout:
         - If goal is reached => rollout_reward = 1.0 + novelty_bonus
         - Else, compute base novelty. If base novelty > threshold_novel => add novelty_bonus
         - Then pick a random action from get_actions(state), 
           do predicted_next_state = simulator.predict_next_state(state, action)
         - Continue until depth or goal.
        
        Final novelty = average of rollout_reward across rollouts.
        """
        # If the environment or simulator is missing, return 0 novelty
        if not self.simulator or not hasattr(self.simulator, "predict_next_state"):
            return 0.0

        # Quick check if the current state itself is a goal
        if self.simulator.is_goal(observation):
            return 10.0 + self.novelty_bonus  # Max novelty if we're already at a goal

        
        total_reward = 0.0
        for _ in range(self.number_rollouts):
            rollout_reward = self._simulate_rollout(
                start_state=observation,
                feature_tuples=feature_tuples,
                atom_probs=atom_probs,
                closed_list=closed_list
            )
            total_reward += rollout_reward

        return total_reward / float(self.number_rollouts)

    def _simulate_rollout(
        self,
        start_state: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        """
        Single rollout simulation from start_state, up to rollout_depth steps.
        Returns a "rollout reward": 1.0 if a goal is reached, plus an optional novelty bonus.
        """
        current_state = start_state
        rollout_reward = 0.0
        for depth in range(self.rollout_depth):
            if self.simulator.is_goal(current_state):
                # If reached goal mid-rollout
                rollout_reward = 5.0 + self.novelty_bonus  # big reward
                return rollout_reward

            # Compute a base novelty measure at the current state
            base_nov_score = self.base_novelty.compute_novelty(
                observation=current_state,
                feature_tuples=feature_tuples,
                atom_probs=atom_probs,
                closed_list=closed_list
            )
            # If the base novelty is above threshold => add novelty_bonus
            if self.base_novelty.is_novel(base_nov_score):
                rollout_reward = max(rollout_reward, self.novelty_bonus)

            # pick a random action
            actions = self.simulator.get_actions(current_state)
            if not actions:  # no actions => dead-end
                break
                
            action = random.choice(actions)
            # use the prediction function
            predicted_next_state = self.simulator.predict_next_state(current_state, action)
            current_state = predicted_next_state

        # If we used up all rollout_depth steps without hitting a goal, 
        # the rollout reward is whatever novelty we accumulated
        return rollout_reward

    def is_novel(self, novelty_value: float) -> bool:
        """
        Decide if the averaged rollout score indicates novelty. 
        By default, we call 'novel' if novelty_value > 0.5 
        (meaning at least half the rollouts discovered novelty or a goal).
        """
        return novelty_value > 0.1


# ------------------------------------------------------------------------- 
# Also using predictions for NoveltyRolloutWBS(Novelty) 
# ------------------------------------------------------------------------- 

class NoveltyRolloutWBS(Novelty):
    """
    'Rollout Width-Based Search Probabilistic' approach, now using predictions.
    For each call to compute_novelty(...), we run multiple random rollouts of length 'rollout_depth'.
    If we reach a goal, that rollout returns a novelty = 1.0.
    Otherwise, we use a base novelty measure to evaluate states along the rollout.
    
    This is similar to eq:npowbs_rollout. 
    The final novelty is the average of the rollout novelties.
    
    The difference from the older snippet is we now call `predict_next_state` 
    instead of a direct `step()` function.
    """

    def __init__(
        self, 
        simulator,
        base_novelty,
        threshold_novel: float = 0.6,
        rollout_depth: int = 5,
        number_rollouts: int = 5,
        seed: Optional[int] = None
    ):
        """
        Args:
            simulator: An object with .predict_next_state(...), .get_actions(...), .is_goal(...)
            base_novelty: Another novelty function instance (like NoveltyNaiveChooseMax3).
            threshold_novel: Novelty threshold used in rollouts.
            rollout_depth: Max length of each rollout.
            number_rollouts: Number of random rollouts to perform.
            seed: For reproducibility in random choices.
        """
        self.simulator = simulator
        self.base_novelty = base_novelty
        self.threshold_novel = threshold_novel
        self.rollout_depth = rollout_depth
        self.number_rollouts = number_rollouts

        if seed is not None:
            random.seed(seed)

    def compute_novelty(
        self,
        observation: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        """
        Perform multiple rollouts from the current state. 
        Each rollout ends if:
          - We reach a goal state, or
          - We exceed rollout_depth steps, or
          - The novelty < threshold_novel (depending on your approach).
        
        We average the final novelty across all rollouts.
        """
        if not self.simulator or not hasattr(self.simulator, "predict_next_state"):
            return 0.0

        # If the state is already a goal, let's consider that maximum novelty = 1.0
        if self.simulator.is_goal(observation):
            return 10.0
        
        
        rollout_novelties = []
        for _ in range(self.number_rollouts):
            rollout_value = self._simulate_rollout(
                start_state=observation,
                feature_tuples=feature_tuples,
                atom_probs=atom_probs,
                closed_list=closed_list
            )
            rollout_novelties.append(rollout_value)

        if len(rollout_novelties) == 0:
            return 0.0
        return sum(rollout_novelties) / float(len(rollout_novelties))

    def _simulate_rollout(
        self,
        start_state: Any,
        feature_tuples: List[Tuple[Any, ...]],
        atom_probs: Dict[Any, float],
        closed_list
    ) -> float:
        current_state = start_state
        for step_i in range(self.rollout_depth):
            if self.simulator.is_goal(current_state):
                # If the rollout found a goal
                return 1.0

            # Evaluate novelty using a base novelty measure 
            base_nov = self.base_novelty.compute_novelty(
                observation=current_state,
                feature_tuples=feature_tuples,
                atom_probs=atom_probs,
                closed_list=closed_list
            )
            if base_nov < self.threshold_novel:
                # If novelty falls below threshold, we end rollout with that novelty
                return base_nov

            
            # pick a random action
            actions = self.simulator.get_actions(current_state)
            if not actions:
                return 0.0
            action = random.choice(actions)

            # predict the next state
            next_state = self.simulator.predict_next_state(current_state, action)
            current_state = next_state

        # If we reached the end of rollout_depth w/o goal
        # The final novelty is whatever the last state's base novelty was
        final_nov = self.base_novelty.compute_novelty(
            observation=current_state,
            feature_tuples=feature_tuples,
            atom_probs=atom_probs,
            closed_list=closed_list
        )
        return final_nov

    def is_novel(self, novelty_value: float) -> bool:
        # By default, let's say a rollout-based novelty is novel if > threshold
        return novelty_value > self.threshold_novel
