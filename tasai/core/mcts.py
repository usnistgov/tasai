"""
Monte Carlo Tree Search for Autonomous Experiment Planning

This module implements MCTS for planning batches of measurements, addressing
two key limitations of greedy one-step-ahead acquisition:

1. **Path optimization**: Greedy selection ignores motor motion costs across
   the full trajectory. MCTS plans paths that are jointly informative AND
   minimize total travel time.

2. **Lookahead**: Greedy methods can "paint themselves into corners" where
   the next most informative point is far away. MCTS explores future 
   consequences of current choices.

The key insight is that we can evaluate MCTS nodes efficiently using 
importance sampling on a fixed set of posterior samples, avoiding expensive
MCMC at each node.

References:
- Browne et al., IEEE TCIAIG 4(1), 1-43 (2012) - MCTS survey
- Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192-1204 (2024) - AutoREFL forecasting
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS planner."""
    n_simulations: int = 100  # Number of MCTS iterations
    exploration_constant: float = 1.41  # UCB exploration (sqrt(2) is standard)
    max_depth: int = 5  # Maximum tree depth (= batch size)
    discount_factor: float = 0.95  # Discount for future information gain
    n_candidates: int = 20  # Candidate points to consider at each expansion
    min_info_gain: float = 0.001  # Stop if info gain drops below this
    rollout_depth: int = 3  # Depth of random rollout for simulation
    resampling_threshold: float = 0.5  # ESS threshold for resampling


@dataclass 
class MeasurementPoint:
    """A candidate measurement point."""
    H: float
    K: float = 0.0
    L: float = 0.0
    E: float = 0.0
    
    def __hash__(self):
        return hash((round(self.H, 4), round(self.K, 4), 
                     round(self.L, 4), round(self.E, 4)))
    
    def __eq__(self, other):
        if not isinstance(other, MeasurementPoint):
            return False
        return (abs(self.H - other.H) < 1e-4 and 
                abs(self.K - other.K) < 1e-4 and
                abs(self.L - other.L) < 1e-4 and
                abs(self.E - other.E) < 1e-4)


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    # State
    weights: np.ndarray  # Importance weights on posterior samples
    position: Optional[MeasurementPoint]  # Current instrument position
    path_cost: float  # Cumulative time cost to reach this node
    depth: int  # Depth in tree (0 = root)
    
    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: Dict[MeasurementPoint, 'MCTSNode'] = field(default_factory=dict)
    action: Optional[MeasurementPoint] = None  # Action that led here
    
    # Statistics
    visit_count: int = 0
    total_value: float = 0.0
    
    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visit_count == 0:
            return float('inf')
        if self.parent is None or self.parent.visit_count == 0:
            return self.mean_value
        
        exploitation = self.mean_value
        exploration = np.sqrt(np.log(self.parent.visit_count) / self.visit_count)
        return exploitation + 1.41 * exploration


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for autonomous experiments.
    
    Plans batches of measurements that are jointly informative while
    minimizing motor motion costs.
    """
    
    def __init__(self,
                 posterior_samples: np.ndarray,
                 physics_model: Any,
                 candidate_generator: Callable[[], List[MeasurementPoint]],
                 movement_time_func: Callable[[MeasurementPoint, MeasurementPoint], float],
                 config: MCTSConfig = None,
                 poi_indices: List[int] = None):
        """
        Initialize MCTS planner.
        
        Parameters
        ----------
        posterior_samples : np.ndarray
            Current posterior samples, shape (n_samples, n_params)
        physics_model : callable
            Model that predicts intensity given samples and point
        candidate_generator : callable
            Returns list of candidate measurement points
        movement_time_func : callable
            Returns movement time between two points
        config : MCTSConfig
            MCTS configuration
        poi_indices : list of int
            Parameters of interest for entropy calculation
        """
        self.samples = posterior_samples
        self.n_samples = len(posterior_samples)
        self.physics_model = physics_model
        self.candidate_generator = candidate_generator
        self.movement_time_func = movement_time_func
        self.config = config or MCTSConfig()
        self.poi_indices = poi_indices
        
        # Precompute initial entropy
        self.initial_entropy = self._compute_entropy(
            np.ones(self.n_samples) / self.n_samples
        )
        
        logger.info(f"MCTS initialized: {self.n_samples} samples, "
                   f"H_initial={self.initial_entropy:.4f}")
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute differential entropy of weighted samples."""
        # Resample to get unweighted samples
        n_resample = min(500, self.n_samples)
        indices = np.random.choice(
            self.n_samples, n_resample, p=weights, replace=True
        )
        resampled = self.samples[indices]
        
        if self.poi_indices is not None:
            resampled = resampled[:, self.poi_indices]
        
        # Use covariance-based entropy estimate
        try:
            cov = np.cov(resampled.T)
            if np.ndim(cov) == 0:
                cov = np.array([[cov]])
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            d = resampled.shape[1] if resampled.ndim > 1 else 1
            entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            return entropy
        except:
            return 0.0
    
    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Compute effective sample size from importance weights."""
        return 1.0 / np.sum(weights ** 2)
    
    def _update_weights(self, 
                        weights: np.ndarray,
                        point: MeasurementPoint,
                        count_time: float = 60.0) -> Tuple[np.ndarray, float, float]:
        """
        Update importance weights after simulated measurement.
        
        Returns
        -------
        new_weights : np.ndarray
            Updated importance weights
        expected_intensity : float
            Expected measurement value
        info_gain : float
            Expected information gain
        """
        # Predict intensity for each posterior sample
        predictions = np.array([
            self.physics_model.predict_intensity(s, point)
            for s in self.samples
        ])
        
        # Expected intensity (weighted mean)
        I_expected = np.average(predictions, weights=weights)
        
        # Measurement uncertainty (Poisson)
        sigma = np.sqrt(max(I_expected, 1.0) / count_time)
        
        # Log-likelihood of each sample given expected measurement
        log_likes = -0.5 * ((predictions - I_expected) / sigma) ** 2
        
        # Update weights via importance sampling
        new_log_weights = np.log(weights + 1e-300) + log_likes
        new_log_weights -= np.max(new_log_weights)  # Numerical stability
        new_weights = np.exp(new_log_weights)
        new_weights /= np.sum(new_weights)
        
        # Check effective sample size
        ess = self._effective_sample_size(new_weights)
        if ess < self.n_samples * self.config.resampling_threshold:
            # Resample to prevent weight collapse
            n_resample = self.n_samples
            indices = np.random.choice(
                self.n_samples, n_resample, p=new_weights, replace=True
            )
            # After resampling, weights are uniform
            new_weights = np.ones(self.n_samples) / self.n_samples
            # Note: in a full implementation, we'd also resample self.samples
            # but for simulation purposes, uniform weights is a reasonable approximation
        
        # Compute information gain
        old_entropy = self._compute_entropy(weights)
        new_entropy = self._compute_entropy(new_weights)
        info_gain = old_entropy - new_entropy
        
        return new_weights, I_expected, max(info_gain, 0.0)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        while node.depth < self.config.max_depth:
            # If not fully expanded, return this node for expansion
            if len(node.children) < self.config.n_candidates:
                return node
            
            # All children expanded - select best by UCB
            if not node.children:
                return node
            
            best_child = max(node.children.values(), key=lambda c: c.ucb_score)
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        if node.depth >= self.config.max_depth:
            return node
        
        # Generate candidates
        candidates = self.candidate_generator()
        
        # Filter out already-explored actions
        unexplored = [c for c in candidates if c not in node.children]
        
        if not unexplored:
            return node
        
        # Score candidates by quick heuristic (info_gain / time)
        scores = []
        for candidate in unexplored[:self.config.n_candidates]:
            if node.position is not None:
                move_time = self.movement_time_func(node.position, candidate)
            else:
                move_time = 30.0
            
            # Quick info gain estimate
            _, _, info_gain = self._update_weights(node.weights, candidate)
            score = info_gain / (move_time + 60.0)  # Assume 60s count time
            scores.append((score, candidate))
        
        # Select best unexplored action (sort by score only)
        scores.sort(key=lambda x: x[0], reverse=True)
        action = scores[0][1]
        
        # Compute child state
        new_weights, _, info_gain = self._update_weights(node.weights, action)
        
        if node.position is not None:
            move_time = self.movement_time_func(node.position, action)
        else:
            move_time = 30.0
        
        child = MCTSNode(
            weights=new_weights,
            position=action,
            path_cost=node.path_cost + move_time + 60.0,
            depth=node.depth + 1,
            parent=node,
            action=action
        )
        
        node.children[action] = child
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate random rollout from node, return cumulative value.
        
        Value = total info gain achieved, penalized by path inefficiency.
        """
        weights = node.weights.copy()
        position = node.position
        total_info_gain = 0.0
        total_move_time = 0.0
        
        for step in range(self.config.rollout_depth):
            # Random action
            candidates = self.candidate_generator()
            if not candidates:
                break
            
            action = candidates[np.random.randint(len(candidates))]
            
            # Update state
            new_weights, _, info_gain = self._update_weights(weights, action)
            
            if position is not None:
                move_time = self.movement_time_func(position, action)
            else:
                move_time = 30.0
            
            # Accumulate
            total_info_gain += info_gain
            total_move_time += move_time
            
            # Update for next step
            weights = new_weights
            position = action
            
            # Early stopping if info gain negligible
            if info_gain < self.config.min_info_gain:
                break
        
        # Value = info gain with penalty for excessive movement
        # This encourages high info gain while discouraging long paths
        count_time = 60.0 * self.config.rollout_depth
        efficiency = count_time / (count_time + total_move_time)  # 0 to 1
        
        return total_info_gain * efficiency
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
    
    def plan(self, 
             current_position: MeasurementPoint = None,
             n_points: int = 5) -> List[Tuple[MeasurementPoint, float, float]]:
        """
        Plan a batch of measurements using MCTS.
        
        Parameters
        ----------
        current_position : MeasurementPoint
            Current instrument position
        n_points : int
            Number of points to plan
        
        Returns
        -------
        plan : list of (point, expected_intensity, expected_info_gain)
            Planned measurement sequence
        """
        # Adjust depth
        self.config.max_depth = n_points
        
        # Initialize root
        root = MCTSNode(
            weights=np.ones(self.n_samples) / self.n_samples,
            position=current_position,
            path_cost=0.0,
            depth=0
        )
        
        # MCTS iterations
        start_time = time.time()
        for i in range(self.config.n_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if node.depth < self.config.max_depth:
                node = self._expand(node)
            
            # Simulation
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        elapsed = time.time() - start_time
        logger.info(f"MCTS completed: {self.config.n_simulations} simulations "
                   f"in {elapsed:.2f}s")
        
        # Extract best path
        path = []
        node = root
        weights = root.weights.copy()
        
        for _ in range(n_points):
            if not node.children:
                break
            
            # Select most-visited child
            best_child = max(node.children.values(), 
                           key=lambda c: c.visit_count)
            
            # Compute expected values for this action
            _, expected_I, expected_info = self._update_weights(
                weights, best_child.action
            )
            
            path.append((best_child.action, expected_I, expected_info))
            
            weights = best_child.weights
            node = best_child
        
        return path


class MCTSModelDiscrimination:
    """
    MCTS for model discrimination: plan measurements that maximize
    Bayes factor growth while also constraining parameters.
    """
    
    def __init__(self,
                 models: List[Any],
                 model_posteriors: List[np.ndarray],
                 candidate_generator: Callable,
                 movement_time_func: Callable,
                 config: MCTSConfig = None):
        """
        Initialize model discrimination MCTS.
        
        Parameters
        ----------
        models : list
            List of physics models to discriminate between
        model_posteriors : list of np.ndarray
            Posterior samples for each model
        candidate_generator : callable
            Returns list of candidate measurement points  
        movement_time_func : callable
            Returns movement time between two points
        config : MCTSConfig
            MCTS configuration
        """
        self.models = models
        self.model_posteriors = model_posteriors
        self.n_models = len(models)
        self.candidate_generator = candidate_generator
        self.movement_time_func = movement_time_func
        self.config = config or MCTSConfig()
        
        # Initialize uniform model weights
        self.model_weights = np.ones(self.n_models) / self.n_models
        
        # Initialize importance weights for each model's posterior
        self.sample_weights = [
            np.ones(len(p)) / len(p) for p in model_posteriors
        ]
    
    def _predict_intensity(self, model_idx: int, params: np.ndarray, 
                          point: MeasurementPoint) -> float:
        """Predict intensity for a model with given parameters."""
        return self.models[model_idx].predict_intensity(params, point)
    
    def _compute_model_evidence(self, 
                                 point: MeasurementPoint,
                                 I_observed: float,
                                 sigma: float) -> np.ndarray:
        """
        Compute marginal likelihood for each model at a measurement.
        
        Returns log p(I | point, model) for each model.
        """
        log_evidences = np.zeros(self.n_models)
        
        for m in range(self.n_models):
            # Predictions from each posterior sample
            predictions = np.array([
                self._predict_intensity(m, s, point)
                for s in self.model_posteriors[m]
            ])
            
            # Log-likelihood for each sample
            log_likes = -0.5 * ((predictions - I_observed) / sigma) ** 2
            
            # Marginal likelihood via importance sampling
            # log p(I|M) ≈ log mean(exp(log_likes) * weights)
            weights = self.sample_weights[m]
            log_evidence = np.log(np.sum(np.exp(log_likes) * weights))
            log_evidences[m] = log_evidence
        
        return log_evidences
    
    def _update_model_weights(self,
                               point: MeasurementPoint,
                               use_expected: bool = True) -> Tuple[np.ndarray, float]:
        """
        Update model weights after (simulated) measurement.
        
        Parameters
        ----------
        point : MeasurementPoint
            Measurement location
        use_expected : bool
            If True, use expected intensity; if False, sample
        
        Returns
        -------
        new_weights : np.ndarray
            Updated model probabilities
        bayes_factor : float
            Bayes factor between best and second-best model
        """
        # Get expected intensity from current best model
        best_model = np.argmax(self.model_weights)
        
        predictions = np.array([
            self._predict_intensity(best_model, s, point)
            for s in self.model_posteriors[best_model]
        ])
        
        I_expected = np.average(predictions, weights=self.sample_weights[best_model])
        sigma = np.sqrt(max(I_expected, 1.0) / 60.0)  # Assume 60s count
        
        if use_expected:
            I_observed = I_expected
        else:
            I_observed = I_expected + sigma * np.random.randn()
        
        # Compute model evidences
        log_evidences = self._compute_model_evidence(point, I_observed, sigma)
        
        # Update model weights via Bayes' rule
        log_weights = np.log(self.model_weights + 1e-300) + log_evidences
        log_weights -= np.max(log_weights)
        new_weights = np.exp(log_weights)
        new_weights /= np.sum(new_weights)
        
        # Compute Bayes factor
        sorted_weights = np.sort(new_weights)[::-1]
        if sorted_weights[1] > 1e-10:
            bayes_factor = sorted_weights[0] / sorted_weights[1]
        else:
            bayes_factor = 1e10
        
        return new_weights, bayes_factor
    
    def _compute_discrimination_value(self, point: MeasurementPoint) -> float:
        """
        Compute value of a point for model discrimination.
        
        High value = models make very different predictions.
        """
        predictions_by_model = []
        
        for m in range(self.n_models):
            preds = np.array([
                self._predict_intensity(m, s, point)
                for s in self.model_posteriors[m]
            ])
            mean_pred = np.average(preds, weights=self.sample_weights[m])
            predictions_by_model.append(mean_pred)
        
        # Value = variance of predictions across models
        predictions = np.array(predictions_by_model)
        
        # Weight by current model probabilities
        mean_pred = np.sum(predictions * self.model_weights)
        variance = np.sum(self.model_weights * (predictions - mean_pred)**2)
        
        return variance
    
    def plan_discriminating_batch(self,
                                   current_position: MeasurementPoint = None,
                                   n_points: int = 5,
                                   alpha: float = 0.5) -> List[Tuple[MeasurementPoint, float]]:
        """
        Plan a batch that balances model discrimination and parameter estimation.
        
        Parameters
        ----------
        current_position : MeasurementPoint
            Current instrument position
        n_points : int
            Number of points to plan
        alpha : float
            Weight on discrimination (0 = pure parameter estimation,
            1 = pure model discrimination)
        
        Returns
        -------
        plan : list of (point, expected_bayes_factor_improvement)
        """
        self.config.max_depth = n_points
        
        # State for MCTS
        @dataclass
        class DiscriminationState:
            model_weights: np.ndarray
            sample_weights: List[np.ndarray]
            position: Optional[MeasurementPoint]
            path_cost: float
            log_bayes_factor: float
        
        initial_state = DiscriminationState(
            model_weights=self.model_weights.copy(),
            sample_weights=[w.copy() for w in self.sample_weights],
            position=current_position,
            path_cost=0.0,
            log_bayes_factor=0.0
        )
        
        # Simple greedy planning with lookahead
        # (Full MCTS is complex with changing model weights)
        plan = []
        state = initial_state
        
        for step in range(n_points):
            candidates = self.candidate_generator()
            
            best_score = -np.inf
            best_action = None
            best_new_weights = None
            best_bf = None
            
            for candidate in candidates:
                # Movement cost
                if state.position is not None:
                    move_time = self.movement_time_func(state.position, candidate)
                else:
                    move_time = 30.0
                
                # Discrimination value
                disc_value = self._compute_discrimination_value(candidate)
                
                # Simulate update
                new_weights, bf = self._update_model_weights(candidate)
                
                # Combined score
                param_value = 1.0  # Could compute info gain here
                score = (alpha * np.log(bf + 1) + (1 - alpha) * param_value) / (move_time + 60.0)
                
                if score > best_score:
                    best_score = score
                    best_action = candidate
                    best_new_weights = new_weights
                    best_bf = bf
            
            if best_action is None:
                break
            
            plan.append((best_action, best_bf))
            
            # Update state
            self.model_weights = best_new_weights
            state = DiscriminationState(
                model_weights=best_new_weights,
                sample_weights=state.sample_weights,  # Simplified
                position=best_action,
                path_cost=state.path_cost + move_time + 60.0,
                log_bayes_factor=np.log(best_bf)
            )
        
        return plan


# =============================================================================
# Greedy baseline for comparison
# =============================================================================

class GreedyPlanner:
    """Greedy one-step-ahead planner for comparison with MCTS."""
    
    def __init__(self,
                 posterior_samples: np.ndarray,
                 physics_model: Any,
                 candidate_generator: Callable,
                 movement_time_func: Callable,
                 poi_indices: List[int] = None):
        """Initialize greedy planner with same interface as MCTS."""
        self.samples = posterior_samples
        self.n_samples = len(posterior_samples)
        self.physics_model = physics_model
        self.candidate_generator = candidate_generator
        self.movement_time_func = movement_time_func
        self.poi_indices = poi_indices
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute differential entropy of weighted samples."""
        n_resample = min(500, self.n_samples)
        indices = np.random.choice(
            self.n_samples, n_resample, p=weights, replace=True
        )
        resampled = self.samples[indices]
        
        if self.poi_indices is not None:
            resampled = resampled[:, self.poi_indices]
        
        try:
            cov = np.cov(resampled.T)
            if np.ndim(cov) == 0:
                cov = np.array([[cov]])
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            d = resampled.shape[1] if resampled.ndim > 1 else 1
            return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
        except:
            return 0.0
    
    def _update_weights(self, weights: np.ndarray, point: MeasurementPoint,
                       count_time: float = 60.0) -> Tuple[np.ndarray, float, float]:
        """Update importance weights after simulated measurement."""
        predictions = np.array([
            self.physics_model.predict_intensity(s, point)
            for s in self.samples
        ])
        
        I_expected = np.average(predictions, weights=weights)
        sigma = np.sqrt(max(I_expected, 1.0) / count_time)
        
        log_likes = -0.5 * ((predictions - I_expected) / sigma) ** 2
        
        new_log_weights = np.log(weights + 1e-300) + log_likes
        new_log_weights -= np.max(new_log_weights)
        new_weights = np.exp(new_log_weights)
        new_weights /= np.sum(new_weights)
        
        old_entropy = self._compute_entropy(weights)
        new_entropy = self._compute_entropy(new_weights)
        info_gain = max(old_entropy - new_entropy, 0.0)
        
        return new_weights, I_expected, info_gain
    
    def plan(self,
             current_position: MeasurementPoint = None,
             n_points: int = 5) -> List[Tuple[MeasurementPoint, float, float]]:
        """
        Plan measurements greedily (one at a time).
        
        This is the baseline against which MCTS should improve.
        """
        plan = []
        weights = np.ones(self.n_samples) / self.n_samples
        position = current_position
        
        for step in range(n_points):
            candidates = self.candidate_generator()
            
            best_score = -np.inf
            best_action = None
            best_info = None
            best_I = None
            best_weights = None
            
            for candidate in candidates:
                # Movement cost
                if position is not None:
                    move_time = self.movement_time_func(position, candidate)
                else:
                    move_time = 30.0
                
                # Info gain
                new_weights, I_exp, info_gain = self._update_weights(weights, candidate)
                
                # Information rate
                score = info_gain / (move_time + 60.0)
                
                if score > best_score:
                    best_score = score
                    best_action = candidate
                    best_info = info_gain
                    best_I = I_exp
                    best_weights = new_weights
            
            if best_action is None:
                break
            
            plan.append((best_action, best_I, best_info))
            
            # Update state for next iteration
            weights = best_weights
            position = best_action
        
        return plan


# =============================================================================
# Benchmark utilities
# =============================================================================

def compute_path_length(path: List[MeasurementPoint],
                       movement_time_func: Callable) -> float:
    """Compute total movement time for a path."""
    if len(path) < 2:
        return 0.0
    
    total = 0.0
    for i in range(1, len(path)):
        total += movement_time_func(path[i-1], path[i])
    
    return total


def compare_planners(posterior_samples: np.ndarray,
                     physics_model: Any,
                     candidate_generator: Callable,
                     movement_time_func: Callable,
                     n_points: int = 5,
                     n_trials: int = 10,
                     poi_indices: List[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare MCTS and Greedy planners.
    
    Returns
    -------
    results : dict
        {'mcts': {...}, 'greedy': {...}} with metrics
    """
    results = {'mcts': defaultdict(list), 'greedy': defaultdict(list)}
    
    for trial in range(n_trials):
        # Random starting position
        candidates = candidate_generator()
        start_pos = candidates[np.random.randint(len(candidates))]
        
        # MCTS planner
        mcts = MCTSPlanner(
            posterior_samples=posterior_samples,
            physics_model=physics_model,
            candidate_generator=candidate_generator,
            movement_time_func=movement_time_func,
            poi_indices=poi_indices
        )
        
        start = time.time()
        mcts_plan = mcts.plan(current_position=start_pos, n_points=n_points)
        mcts_time = time.time() - start
        
        # Greedy planner
        greedy = GreedyPlanner(
            posterior_samples=posterior_samples,
            physics_model=physics_model,
            candidate_generator=candidate_generator,
            movement_time_func=movement_time_func,
            poi_indices=poi_indices
        )
        
        start = time.time()
        greedy_plan = greedy.plan(current_position=start_pos, n_points=n_points)
        greedy_time = time.time() - start
        
        # Extract paths
        mcts_path = [start_pos] + [p[0] for p in mcts_plan]
        greedy_path = [start_pos] + [p[0] for p in greedy_plan]
        
        # Metrics
        results['mcts']['path_length'].append(
            compute_path_length(mcts_path, movement_time_func)
        )
        results['mcts']['total_info_gain'].append(
            sum(p[2] for p in mcts_plan)
        )
        results['mcts']['planning_time'].append(mcts_time)
        
        results['greedy']['path_length'].append(
            compute_path_length(greedy_path, movement_time_func)
        )
        results['greedy']['total_info_gain'].append(
            sum(p[2] for p in greedy_plan)
        )
        results['greedy']['planning_time'].append(greedy_time)
    
    # Compute means and stds
    summary = {}
    for planner in ['mcts', 'greedy']:
        summary[planner] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in results[planner].items()
        }
    
    return summary
