import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from scipy.stats import beta
from scipy.integrate import quad

class ClosedList(ABC):
    @abstractmethod
    def get_probability(self, fluent: Any) -> float:
        pass

    @abstractmethod
    def update(self, state: Any, atom_probs: Dict[Any, float]):
        """
        Update the closed list based on the current observation (state and its atom probabilities).
        This update is typically done after evaluating novelty.
        """
        pass


class MaxClosedList(ClosedList):
    """
    Implements:
    closed_list_max_{i+1}(f) = max(closed_list_max_i(f), sensor_model(state, f))
    """
    def __init__(self):
        self.probs = {}

    def get_probability(self, fluent: Any) -> float:
        return self.probs.get(fluent, 0.0)

    def update(self, state: Any, atom_probs: Dict[Any, float]):
        for f, p in atom_probs.items():
            # closed_list_max_{i+1}(f) = max(old_val, p)
            old_val = self.get_probability(f)
            self.probs[f] = max(old_val, p)


class ProbabilisticClosedList(ClosedList):
    """
    Implements:
    closed_list_prob_{i+1}(f) = 1 - [1 - sensor_model(state, f)] * [1 - closed_list_prob_i(f)]
    """
    def __init__(self):
        self.probs = {}

    def get_probability(self, fluent: Any) -> float:
        return self.probs.get(fluent, 0.0)

    def update(self, state: Any, atom_probs: Dict[Any, float]):
        for f, p in atom_probs.items():
            old_val = self.get_probability(f)
            # new_val = 1 - (1 - p)*(1 - old_val)
            new_val = 1.0 - (1.0 - p)*(1.0 - old_val)
            self.probs[f] = new_val


class BayesianClosedList(ClosedList):
    """
    Implements the Bayesian update:
    closed_list_bayes_{i+1} depends on observation and sensor model.
    See eq:bayes_likelihood_updated

    Requires parameters:
      - true_positive_rate: Probability of observing f if it is in the state.
      - false_positive_rate: Probability of observing f if it is not in the state.
      - threshold_novel: Minimum probability above which we consider updating.
    """
    def __init__(self, true_positive_rate=0.9, false_positive_rate=0.1, threshold_novel=0.1):
        self.probs = {}
        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate
        self.threshold_novel = threshold_novel

    def get_probability(self, fluent: Any) -> float:
        return self.probs.get(fluent, 0.0)

    def update(self, state: Any, atom_probs: Dict[Any, float]):
        for f, p in atom_probs.items():
            old_val = self.get_probability(f)
            # Update only if sensor model probability > threshold_novel
            if p > self.threshold_novel:
                # Bayesian update:
                # posterior = old_val * [ (TP * p) / (TP*p + FP*(1-p)) ]
                denominator = self.true_positive_rate*p + self.false_positive_rate*(1.0 - p)
                if denominator > 0:
                    posterior = old_val * ((self.true_positive_rate*p)/denominator)
                    # Since old_val is prior that fluent appeared before,
                    # To ensure monotonic increase, we can also max with old_val:
                    posterior = max(old_val, posterior)
                    self.probs[f] = posterior
                else:
                    # Degenerate case (should not happen if rates are sensible)
                    self.probs[f] = old_val
            else:
                # If below threshold, no update
                # Just keep old_val
                self.probs[f] = old_val


class ClosedListBetaDistribution(ClosedList):
    """
    Implements the Closed List using Beta distributions to model the probability of each fluent.
    This approach captures both the mean probability and the uncertainty (variance) associated with each fluent.
    """

    def __init__(
        self,
        false_positive_rate: float = 0.1,
        false_negative_rate: float = 0.1,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0
    ):
        """
        Args:
            false_positive_rate (float): Rate of falsely identifying a fluent as present.
            false_negative_rate (float): Rate of failing to identify a present fluent.
            initial_alpha (float): Initial alpha parameter for Beta distribution.
            initial_beta (float): Initial beta parameter for Beta distribution.
        """
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.closed_list_params: Dict[Any, Tuple[float, float]] = {}
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

    def beta_params_from_mean_variance(self, mu: float, variance: float) -> Tuple[float, float]:
        """
        Given a mean and variance, compute the alpha and beta parameters of the Beta distribution.
        """
        # Prevent division by zero
        if variance <= 0:
            return self.initial_alpha, self.initial_beta

        # Calculate alpha + beta
        ab = (mu * (1 - mu)) / variance - 1
        if ab <= 0:
            # Fallback to initial parameters if variance is too low
            return self.initial_alpha, self.initial_beta

        alpha = mu * ab
        beta_ = (1 - mu) * ab

        # Ensure alpha and beta are positive
        alpha = max(alpha, 1e-4)
        beta_ = max(beta_, 1e-4)

        return alpha, beta_

    def initialize_fluent(self, fluent: Any, mu: float):
        """
        Initialize a fluent's Beta distribution parameters based on the observed mean.
        """
        variance = mu * (1 - mu) * ((self.false_positive_rate + self.false_negative_rate) / 2.0)
        alpha, beta_ = self.beta_params_from_mean_variance(mu, variance)
        self.closed_list_params[fluent] = (alpha, beta_)

    def update_fluent(self, fluent: Any, mu_obs: float):
        """
        Update the Beta distribution parameters for a fluent based on a new observation.
        """
        if fluent not in self.closed_list_params:
            # Initialize if not present
            self.initialize_fluent(fluent, mu_obs)
            return

        alpha_closed, beta_closed = self.closed_list_params[fluent]

        # Compute variance based on FPR and FNR
        variance_obs = mu_obs * (1 - mu_obs) * ((self.false_positive_rate + self.false_negative_rate) / 2.0)

        if variance_obs <= 0:
            # Fallback to initial parameters if variance is invalid
            alpha_obs, beta_obs = self.initial_alpha, self.initial_beta
        else:
            # Compute alpha and beta for the observation
            ab_obs = (mu_obs * (1 - mu_obs)) / variance_obs - 1.0
            if ab_obs <= 0:
                ab_obs = 2.0  # Minimal scale to prevent too low precision
            alpha_obs = mu_obs * ab_obs
            beta_obs = (1 - mu_obs) * ab_obs

            # Ensure alpha and beta are positive
            alpha_obs = max(alpha_obs, 1e-4)
            beta_obs = max(beta_obs, 1e-4)

        # Bayesian update: assuming prior is Beta(alpha_closed, beta_closed) and likelihood is Beta(alpha_obs, beta_obs)
        # The conjugate prior property allows us to update parameters by simple addition
        alpha_new = alpha_closed + alpha_obs
        beta_new = beta_closed + beta_obs

        self.closed_list_params[fluent] = (alpha_new, beta_new)

    def get_probability(self, fluent: Any) -> float:
        """
        Retrieve the mean probability of a fluent being in the closed list.
        """
        if fluent not in self.closed_list_params:
            return 0.0  # If not observed before, probability is 0

        alpha, beta_ = self.closed_list_params[fluent]
        return alpha / (alpha + beta_)

    def update(self, state: Any, atom_probs: Dict[Any, float]):
        """
        Update the closed list based on the current observation.
        """
        for fluent, mu_obs in atom_probs.items():
            if fluent not in self.closed_list_params:
                self.initialize_fluent(fluent, mu_obs)
            else:
                self.update_fluent(fluent, mu_obs)

    def get_parameters(self, fluent: Any) -> Optional[Tuple[float, float]]:
        """
        Retrieve the alpha and beta parameters of the Beta distribution for a fluent.
        """
        return self.closed_list_params.get(fluent, None)

    def get_cdf(self, fluent: Any, x: float) -> float:
        """
        Retrieve the cumulative distribution function value at x for a fluent's Beta distribution.
        """
        if fluent not in self.closed_list_params:
            return 0.0
        alpha, beta_ = self.closed_list_params[fluent]
        return beta.cdf(x, alpha, beta_)

    def get_pdf(self, fluent: Any, x: float) -> float:
        """
        Retrieve the probability density function value at x for a fluent's Beta distribution.
        """
        if fluent not in self.closed_list_params:
            return 0.0
        alpha, beta_ = self.closed_list_params[fluent]
        return beta.pdf(x, alpha, beta_)

    def __str__(self):
        """
        String representation for debugging.
        """
        return f"ClosedListBetaDistribution with {len(self.closed_list_params)} fluents."


# Example Usage
if __name__ == "__main__":
    # Initialize Closed List with Beta Distributions
    closed_list = ClosedListBetaDistribution(false_positive_rate=0.05, false_negative_rate=0.05)

    # Example state and atom probabilities
    state = "state_1"
    atom_probs = {
        "On(A,B)": 0.9,
        "Clear(B)": 0.8,
        "Holding(A)": 0.2
    }

    # Update the closed list with the current observation
    closed_list.update(state, atom_probs)

    # Retrieve probabilities
    for fluent in atom_probs.keys():
        prob = closed_list.get_probability(fluent)
        print(f"Fluent: {fluent}, Probability in Closed List: {prob:.4f}")

    # Retrieve Beta parameters
    for fluent in atom_probs.keys():
        params = closed_list.get_parameters(fluent)
        if params:
            print(f"Fluent: {fluent}, Alpha: {params[0]:.4f}, Beta: {params[1]:.4f}")

    # Example of integrating the Beta PDF
    fluent = "On(A,B)"
    I = 0.5  # Example integration point
    integral_val, _ = quad(lambda x: closed_list.get_pdf(fluent, x), I, 1)
    print(f"Integral of PDF from {I} to 1 for {fluent}: {integral_val:.4f}")
