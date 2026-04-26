"""
Monte Carlo Tree Search for Batch Measurement Planning

This module implements MCTS-based planning for selecting batches of measurements
that are jointly informative while minimizing motor motion overhead. It integrates
with the optimistic pipeline (forecasting) from AutoREFL.

The key insight is that greedy one-step-ahead acquisition is suboptimal when:
1. Motor motion costs are significant (traveling salesman aspect)
2. Multiple measurements will be taken before re-running MCMC
3. Measurements interact (information from point A affects value of point B)

MCTS naturally handles all three by planning full trajectories and evaluating
their cumulative information gain per unit time.

References:
- Browne et al., IEEE TCIAIG 4(1), 1-43 (2012) - MCTS survey
- Silver et al., Nature 529, 484-489 (2016) - AlphaGo (MCTS + neural networks)
- Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192-1204 (2024) - AutoREFL forecasting
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS planner."""
    n_simulations: int = 100          # Number of MCTS simulations
    exploration_constant: float = 1.4  # UCB exploration weight (sqrt(2) is theoretically optimal)
    max_depth: int = 10               # Maximum tree depth
    n_candidates: int = 20            # Candidate points to consider at each expansion
    discount_factor: float = 0.95     # Discount for future information gain
    min_visit_count: int = 5          # Minimum visits before expansion
    temperature: float = 1.0          # Temperature for action selection (lower = more greedy)
    use_progressive_widening: bool = True  # Gradually expand action space
    pw_alpha: float = 0.5             # Progressive widening exponent
    allow_dwell: bool = True          # Allow zero-move stacking actions
    dwell_multiplier: float = 3.0     # Count-time multiplier when dwelling
    

@dataclass 
class MeasurementPoint:
    """A point in (Q, E) space."""
    h: float = 0.0
    k: float = 0.0
    l: float = 0.0
    E: float = 0.0
    count_time: float = 60.0
    is_dwell: bool = False
    
    def __hash__(self):
        return hash((
            round(self.h, 4),
            round(self.k, 4),
            round(self.l, 4),
            round(self.E, 4),
            round(self.count_time, 2),
            self.is_dwell
        ))
    
    def __eq__(self, other):
        if not isinstance(other, MeasurementPoint):
            return False
        return (
            abs(self.h - other.h) < 1e-4 and 
            abs(self.k - other.k) < 1e-4 and
            abs(self.l - other.l) < 1e-4 and
            abs(self.E - other.E) < 1e-4 and
            abs(self.count_time - other.count_time) < 1e-2 and
            self.is_dwell == other.is_dwell
        )


class MCTSNode:
    """
    A node in the MCTS tree.
    
    Each node represents a state: (posterior_weights, current_position, path_so_far).
    The posterior samples are fixed from the root; only weights change via importance sampling.
    """
    
    def __init__(self, 
                 posterior_samples: np.ndarray,
                 weights: np.ndarray,
                 position: Optional[MeasurementPoint],
                 path: List[MeasurementPoint],
                 path_time: float,
                 path_info_gain: float,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[MeasurementPoint] = None):
        """
        Initialize MCTS node.
        
        Parameters
        ----------
        posterior_samples : np.ndarray
            Fixed posterior samples from root, shape (n_samples, n_params)
        weights : np.ndarray
            Current importance weights for samples
        position : MeasurementPoint or None
            Current instrument position
        path : list of MeasurementPoint
            Measurements taken to reach this node
        path_time : float
            Total time (count + move) to reach this node
        path_info_gain : float
            Cumulative information gain along path
        parent : MCTSNode or None
            Parent node
        action : MeasurementPoint or None
            Action (measurement) that led to this node
        """
        self.posterior_samples = posterior_samples
        self.weights = weights
        self.position = position
        self.path = path
        self.path_time = path_time
        self.path_info_gain = path_info_gain
        self.parent = parent
        self.action = action
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[MeasurementPoint, 'MCTSNode'] = {}
        
        # Cache entropy for this node
        self._entropy = None
    
    @property
    def entropy(self) -> float:
        """Compute entropy of current (reweighted) posterior."""
        if self._entropy is None:
            self._entropy = self._compute_entropy()
        return self._entropy
    
    def _compute_entropy(self) -> float:
        """Compute differential entropy via resampling."""
        # Ensure weights match samples
        n_samples = len(self.posterior_samples)
        if len(self.weights) != n_samples:
            weights = np.ones(n_samples) / n_samples
        else:
            weights = self.weights
        
        # Normalize weights
        weights = np.asarray(weights).flatten()
        weights = weights / (weights.sum() + 1e-300)
        
        # Resample according to weights
        n_resample = min(n_samples, 500)
        try:
            indices = np.random.choice(
                n_samples, 
                n_resample, 
                p=weights, 
                replace=True
            )
            resampled = self.posterior_samples[indices]
        except ValueError:
            # Fallback to uniform sampling
            indices = np.random.choice(n_samples, n_resample, replace=True)
            resampled = self.posterior_samples[indices]
        
        # Differential entropy via covariance determinant
        try:
            cov = np.cov(resampled.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            d = resampled.shape[1] if resampled.ndim > 1 else 1
            entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            return max(0.0, entropy)
        except:
            return 0.0
    
    @property
    def value(self) -> float:
        """Average value (information rate) from simulations through this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, exploration_constant: float) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def is_fully_expanded(self, n_candidates: int, config: MCTSConfig) -> bool:
        """Check if node is fully expanded (all children visited)."""
        if config.use_progressive_widening:
            # Progressive widening: allow more children as visits increase
            max_children = int(self.visit_count ** config.pw_alpha)
            return len(self.children) >= max_children
        else:
            return len(self.children) >= n_candidates
    
    def is_terminal(self, max_depth: int) -> bool:
        """Check if node is terminal (max depth reached)."""
        return len(self.path) >= max_depth


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for batch measurement selection.
    
    Integrates with the optimistic pipeline by:
    1. Taking posterior samples from MCMC
    2. Planning a batch of N measurements via tree search
    3. Returning the batch for execution while MCMC runs in background
    
    The key advantage over greedy forecasting is handling of:
    - Path-dependent motor motion costs
    - Interaction between measurements (information saturation)
    - Long-horizon planning
    """
    
    def __init__(self,
                 physics_model: Any,
                 candidate_generator: Callable[[], List[MeasurementPoint]],
                 motor_model: Callable[[MeasurementPoint, MeasurementPoint], float],
                 config: Optional[MCTSConfig] = None,
                 poi_indices: Optional[List[int]] = None):
        """
        Initialize MCTS planner.
        
        Parameters
        ----------
        physics_model : PhysicsModel
            Model for predicting intensities given parameters
        candidate_generator : callable
            Function returning list of candidate measurement points
        motor_model : callable
            Function(from_pos, to_pos) -> time in seconds
        config : MCTSConfig, optional
            MCTS configuration
        poi_indices : list of int, optional
            Indices of parameters of interest for entropy calculation
        """
        self.physics_model = physics_model
        self.candidate_generator = candidate_generator
        self.motor_model = motor_model
        self.config = config or MCTSConfig()
        self.poi_indices = poi_indices
        
        # Statistics
        self.stats = defaultdict(float)
    
    def plan_batch(self,
                   posterior_samples: np.ndarray,
                   n_points: int = 5,
                   current_position: Optional[MeasurementPoint] = None,
                   root_entropy: Optional[float] = None) -> List[MeasurementPoint]:
        """
        Plan a batch of measurements using MCTS.
        
        Parameters
        ----------
        posterior_samples : np.ndarray
            Current posterior samples, shape (n_samples, n_params)
        n_points : int
            Number of points to plan
        current_position : MeasurementPoint, optional
            Current instrument position
        root_entropy : float, optional
            Entropy at root (computed if not provided)
        
        Returns
        -------
        list of MeasurementPoint
            Planned measurement batch
        """
        # Extract POI if specified
        if self.poi_indices is not None:
            samples = posterior_samples[:, self.poi_indices]
        else:
            samples = posterior_samples
        
        # Initialize root
        weights = np.ones(len(samples)) / len(samples)
        root = MCTSNode(
            posterior_samples=samples,
            weights=weights,
            position=current_position,
            path=[],
            path_time=0.0,
            path_info_gain=0.0
        )
        
        if root_entropy is None:
            root_entropy = root.entropy
        
        self.root_entropy = root_entropy
        
        logger.info(f"Starting MCTS: {self.config.n_simulations} simulations, "
                   f"planning {n_points} points, H_root={root_entropy:.3f}")
        
        # Run MCTS simulations
        for i in range(self.config.n_simulations):
            self._simulate(root, n_points)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"MCTS simulation {i+1}/{self.config.n_simulations}, "
                           f"root visits={root.visit_count}")
        
        # Extract best path
        best_path = self._extract_best_path(root, n_points)
        
        # Log statistics
        total_info_gain = self._compute_path_info_gain(root, best_path)
        total_time = self._compute_path_time(current_position, best_path)
        
        logger.info(f"MCTS complete: {len(best_path)} points planned, "
                   f"expected ΔH={total_info_gain:.4f}, time={total_time:.1f}s, "
                   f"rate={total_info_gain/max(total_time, 1):.6f}")
        
        self.stats['n_simulations'] = self.config.n_simulations
        self.stats['root_visits'] = root.visit_count
        self.stats['expected_info_gain'] = total_info_gain
        self.stats['expected_time'] = total_time
        
        return best_path
    
    def _simulate(self, root: MCTSNode, max_depth: int):
        """Run one MCTS simulation: select → expand → rollout → backprop."""
        node = root
        
        # Selection: traverse tree using UCB
        while not node.is_terminal(max_depth) and node.is_fully_expanded(
            self.config.n_candidates, self.config
        ):
            child = self._select_child(node)
            if child is None:
                break
            node = child
        
        # Expansion: add new child if not terminal
        if not node.is_terminal(max_depth):
            expanded = self._expand(node)
            if expanded is not None:
                node = expanded
        
        # Rollout: estimate value via random simulation
        value = self._rollout(node, max_depth)
        
        # Backpropagation: update statistics
        self._backpropagate(node, value)
    
    def _select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select child with highest UCB score."""
        if not node.children:
            return None
            
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(self.config.exploration_constant)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        # Generate candidates
        candidates = list(self.candidate_generator())
        if (self.config.allow_dwell and node.position is not None):
            dwell_point = MeasurementPoint(
                h=node.position.h,
                k=node.position.k,
                l=node.position.l,
                E=node.position.E,
                count_time=node.position.count_time * self.config.dwell_multiplier,
                is_dwell=True
            )
            candidates.append(dwell_point)
        
        # Filter out already-expanded actions
        available = [c for c in candidates if c not in node.children]
        
        if not available:
            # All candidates expanded, return random existing child
            if node.children:
                return np.random.choice(list(node.children.values()))
            return node
        
        # Score candidates by expected immediate value
        scores = []
        for candidate in available:
            score = self._score_candidate(node, candidate)
            scores.append(score)
        
        # Select candidate (weighted by score for diversity)
        scores = np.array(scores)
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = np.ones(len(available)) / len(available)
        
        action = available[np.random.choice(len(available), p=probs)]
        
        # Create child node
        child = self._create_child(node, action)
        node.children[action] = child
        
        return child
    
    def _score_candidate(self, node: MCTSNode, candidate: MeasurementPoint) -> float:
        """Score a candidate action for expansion selection."""
        # Predict intensity spread
        predictions = self._predict_intensities(node.posterior_samples, candidate)
        I_std = np.std(predictions)
        
        # Higher variance = more informative
        # Lower move time = more efficient
        move_time = self._get_move_time(node.position, candidate)
        
        time_scale = max(candidate.count_time, 1e-3)
        integration_boost = np.sqrt(time_scale / 60.0)
        score = (I_std * integration_boost) / (time_scale + move_time + 1)
        return max(score, 1e-10)
    
    def _create_child(self, parent: MCTSNode, action: MeasurementPoint) -> MCTSNode:
        """Create child node by simulating measurement."""
        # Predict intensities
        predictions = self._predict_intensities(parent.posterior_samples, action)
        predictions = np.asarray(predictions).flatten()
        
        # Ensure weights match predictions length
        weights = parent.weights
        if len(weights) != len(predictions):
            weights = np.ones(len(predictions)) / len(predictions)
        
        # Expected measurement (posterior mean)
        I_expected = np.average(predictions, weights=weights)
        
        # Measurement uncertainty (Poisson)
        sigma = np.sqrt(max(I_expected, 1) / action.count_time)
        
        # Importance sampling update
        log_likes = -0.5 * ((predictions - I_expected) / max(sigma, 0.1)) ** 2
        new_log_weights = np.log(weights + 1e-300) + log_likes
        new_log_weights -= np.max(new_log_weights)
        new_weights = np.exp(new_log_weights)
        new_weights /= np.sum(new_weights)
        
        # Compute times
        move_time = self._get_move_time(parent.position, action)
        new_path_time = parent.path_time + move_time + action.count_time
        
        # Create child
        child = MCTSNode(
            posterior_samples=parent.posterior_samples,
            weights=new_weights,
            position=action,
            path=parent.path + [action],
            path_time=new_path_time,
            path_info_gain=0.0,  # Computed lazily
            parent=parent,
            action=action
        )
        
        # Compute info gain
        child.path_info_gain = self.root_entropy - child.entropy
        
        return child
    
    def _rollout(self, node: MCTSNode, max_depth: int) -> float:
        """
        Estimate value via random rollout from node.
        
        Returns information rate (info_gain / time) for the full path.
        """
        if node.is_terminal(max_depth):
            # Terminal: return final information rate
            if node.path_time > 0:
                return node.path_info_gain / node.path_time
            return 0.0
        
        # Simulate random continuation
        n_samples = len(node.posterior_samples)
        current_weights = np.ones(n_samples) / n_samples  # Start with uniform
        current_entropy = node.entropy
        current_position = node.position
        current_time = node.path_time
        total_info_gain = node.path_info_gain
        
        depth = len(node.path)
        
        while depth < max_depth:
            # Pick random candidate (including dwell option if applicable)
            candidates = list(self.candidate_generator())
            if self.config.allow_dwell and current_position is not None:
                dwell_point = MeasurementPoint(
                    h=current_position.h,
                    k=current_position.k,
                    l=current_position.l,
                    E=current_position.E,
                    count_time=current_position.count_time * self.config.dwell_multiplier,
                    is_dwell=True
                )
                candidates.append(dwell_point)
            if not candidates:
                break
            
            action = candidates[np.random.randint(len(candidates))]
            
            # Simulate measurement
            predictions = self._predict_intensities(node.posterior_samples, action)
            predictions = np.asarray(predictions).flatten()
            
            # Ensure weights match
            if len(predictions) != len(current_weights):
                current_weights = np.ones(len(predictions)) / len(predictions)
            
            I_expected = np.average(predictions, weights=current_weights)
            sigma = max(np.sqrt(max(I_expected, 1) / action.count_time), 0.1)
            
            # Update weights
            log_likes = -0.5 * ((predictions - I_expected) / sigma) ** 2
            new_log_weights = np.log(current_weights + 1e-300) + log_likes
            new_log_weights -= np.max(new_log_weights)
            current_weights = np.exp(new_log_weights)
            current_weights /= (np.sum(current_weights) + 1e-300)
            
            # Update time
            move_time = self._get_move_time(current_position, action)
            current_time += move_time + action.count_time
            current_position = action
            
            depth += 1
        
        # Compute final entropy and info gain
        n_resample = min(len(node.posterior_samples), 200)
        try:
            # Normalize weights
            current_weights = current_weights / (current_weights.sum() + 1e-300)
            indices = np.random.choice(
                len(node.posterior_samples), n_resample, 
                p=current_weights, replace=True
            )
            resampled = node.posterior_samples[indices]
        except ValueError:
            indices = np.random.choice(len(node.posterior_samples), n_resample, replace=True)
            resampled = node.posterior_samples[indices]
        
        try:
            cov = np.cov(resampled.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            sign, logdet = np.linalg.slogdet(cov)
            if sign > 0:
                d = resampled.shape[1] if resampled.ndim > 1 else 1
                final_entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
                final_entropy = max(0.0, final_entropy)
            else:
                final_entropy = 0.0
        except:
            final_entropy = current_entropy
        
        total_info_gain = self.root_entropy - final_entropy
        
        # Return information rate
        if current_time > 0:
            return total_info_gain / current_time
        return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        discount = 1.0
        while node is not None:
            node.visit_count += 1
            node.total_value += value * discount
            discount *= self.config.discount_factor
            node = node.parent
    
    def _extract_best_path(self, root: MCTSNode, n_points: int) -> List[MeasurementPoint]:
        """Extract best path from tree by following most-visited children."""
        path = []
        node = root
        
        for _ in range(n_points):
            if not node.children:
                break
            
            # Select most-visited child (robust choice)
            # With temperature for exploration
            visits = np.array([c.visit_count for c in node.children.values()])
            
            if self.config.temperature == 0:
                # Greedy
                best_idx = np.argmax(visits)
            else:
                # Softmax with temperature
                logits = visits / max(self.config.temperature, 0.01)
                probs = np.exp(logits - np.max(logits))
                probs /= probs.sum()
                best_idx = np.random.choice(len(visits), p=probs)
            
            best_child = list(node.children.values())[best_idx]
            path.append(best_child.action)
            node = best_child
        
        return path
    
    def _predict_intensities(self, samples: np.ndarray, point: MeasurementPoint) -> np.ndarray:
        """Predict intensity for each posterior sample at given point."""
        predictions = []
        for params in samples:
            try:
                intensity = self.physics_model.intensity(
                    point.h, point.k, point.l, point.E, params
                )
                predictions.append(intensity)
            except:
                predictions.append(0.0)
        return np.array(predictions)
    
    def _get_move_time(self, 
                       from_pos: Optional[MeasurementPoint], 
                       to_pos: MeasurementPoint) -> float:
        """Get motor motion time between positions."""
        if from_pos is None:
            return 30.0  # Default startup time
        if to_pos.is_dwell and from_pos is not None:
            # Staying put to integrate longer: zero move cost
            return 0.0
        return self.motor_model(from_pos, to_pos)
    
    def _compute_path_info_gain(self, root: MCTSNode, path: List[MeasurementPoint]) -> float:
        """Compute expected information gain for a path."""
        if not path:
            return 0.0
        
        # Traverse tree to find info gain
        node = root
        for action in path:
            if action in node.children:
                node = node.children[action]
            else:
                break
        
        return node.path_info_gain if hasattr(node, 'path_info_gain') else 0.0
    
    def _compute_path_time(self, 
                          start: Optional[MeasurementPoint], 
                          path: List[MeasurementPoint]) -> float:
        """Compute total time for a path."""
        if not path:
            return 0.0
        
        total = 0.0
        current = start
        
        for point in path:
            total += self._get_move_time(current, point) + point.count_time
            current = point
        
        return total


# =============================================================================
# Comparison with Greedy Forecasting
# =============================================================================

def compare_mcts_vs_greedy(
    posterior_samples: np.ndarray,
    physics_model: Any,
    candidate_generator: Callable,
    motor_model: Callable,
    n_points: int = 5,
    current_position: Optional[MeasurementPoint] = None,
    n_trials: int = 10
) -> Dict[str, Any]:
    """
    Compare MCTS planning with greedy forecasting.
    
    Returns statistics on information rate improvement.
    """
    from ..core.forecast import Forecaster
    
    results = {
        'mcts_info_rates': [],
        'greedy_info_rates': [],
        'mcts_times': [],
        'greedy_times': []
    }
    
    # MCTS planner
    mcts = MCTSPlanner(
        physics_model=physics_model,
        candidate_generator=candidate_generator,
        motor_model=motor_model,
        config=MCTSConfig(n_simulations=100)
    )
    
    # Greedy forecaster
    greedy = Forecaster(
        physics_model=physics_model,
        candidate_generator=candidate_generator,
        movement_time_func=motor_model
    )
    
    for trial in range(n_trials):
        # MCTS
        mcts_path = mcts.plan_batch(
            posterior_samples, n_points, current_position
        )
        results['mcts_info_rates'].append(
            mcts.stats['expected_info_gain'] / max(mcts.stats['expected_time'], 1)
        )
        results['mcts_times'].append(mcts.stats['expected_time'])
        
        # Greedy
        greedy_result = greedy.forecast(
            posterior_samples, n_points, current_position
        )
        greedy_time = sum(fp.count_time for fp in greedy_result.points)
        greedy_info = greedy_result.total_info_gain
        results['greedy_info_rates'].append(greedy_info / max(greedy_time, 1))
        results['greedy_times'].append(greedy_time)
    
    # Summary
    mcts_rate = np.mean(results['mcts_info_rates'])
    greedy_rate = np.mean(results['greedy_info_rates'])
    improvement = (mcts_rate - greedy_rate) / greedy_rate * 100 if greedy_rate > 0 else 0
    
    logger.info(f"MCTS vs Greedy comparison ({n_trials} trials):")
    logger.info(f"  MCTS info rate: {mcts_rate:.6f} ± {np.std(results['mcts_info_rates']):.6f}")
    logger.info(f"  Greedy info rate: {greedy_rate:.6f} ± {np.std(results['greedy_info_rates']):.6f}")
    logger.info(f"  Improvement: {improvement:.1f}%")
    
    results['mcts_mean_rate'] = mcts_rate
    results['greedy_mean_rate'] = greedy_rate
    results['improvement_percent'] = improvement
    
    return results


# =============================================================================
# Integration with Optimistic Pipeline
# =============================================================================

class OptimisticMCTSPipeline:
    """
    Combines MCTS planning with the optimistic pipeline from AutoREFL.
    
    The pipeline:
    1. MCMC inference (slow, ~500ms)
    2. MCTS plans batch of N measurements
    3. Execute measurements while MCMC runs in background
    4. Repeat
    
    This keeps the instrument continuously measuring while planning happens
    in parallel.
    """
    
    def __init__(self,
                 physics_model: Any,
                 instrument: Any,
                 candidate_generator: Callable,
                 motor_model: Callable,
                 mcmc_runner: Any,
                 mcts_config: Optional[MCTSConfig] = None,
                 batch_size: int = 5):
        """
        Initialize optimistic MCTS pipeline.
        
        Parameters
        ----------
        physics_model : PhysicsModel
            Physics model for predictions
        instrument : Instrument
            Instrument interface for measurements
        candidate_generator : callable
            Candidate point generator
        motor_model : callable
            Motor motion time model
        mcmc_runner : MCMCRunner
            MCMC inference runner
        mcts_config : MCTSConfig, optional
            MCTS configuration
        batch_size : int
            Number of points to plan per batch
        """
        self.physics_model = physics_model
        self.instrument = instrument
        self.candidate_generator = candidate_generator
        self.motor_model = motor_model
        self.mcmc_runner = mcmc_runner
        self.batch_size = batch_size
        
        self.mcts = MCTSPlanner(
            physics_model=physics_model,
            candidate_generator=candidate_generator,
            motor_model=motor_model,
            config=mcts_config or MCTSConfig()
        )
        
        # Measurement history
        self.measurements = []
        self.posterior_samples = None
        self.current_position = None
    
    def run_iteration(self) -> List[Dict]:
        """
        Run one iteration of the pipeline.
        
        Returns list of measurement results.
        """
        # Run MCMC to get posterior
        logger.info("Running MCMC inference...")
        self.posterior_samples = self.mcmc_runner.run(self.measurements)
        
        # Plan batch with MCTS
        logger.info(f"Planning batch of {self.batch_size} measurements with MCTS...")
        batch = self.mcts.plan_batch(
            self.posterior_samples,
            n_points=self.batch_size,
            current_position=self.current_position
        )
        
        # Execute measurements
        results = []
        for point in batch:
            logger.info(f"Measuring at H={point.h:.3f}, E={point.E:.2f}")
            
            result = self.instrument.measure(point)
            
            self.measurements.append({
                'point': point,
                'intensity': result.intensity,
                'uncertainty': result.uncertainty
            })
            
            results.append({
                'h': point.h, 'k': point.k, 'l': point.l, 'E': point.E,
                'intensity': result.intensity,
                'uncertainty': result.uncertainty
            })
            
            self.current_position = point
        
        return results
    
    def run(self,
            max_iterations: int = None,
            max_measurements: int = None,
            stopping_criterion: Callable = None,
            initial_measurements: List[Dict] = None) -> Dict:
        """
        Run full autonomous experiment.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum MCMC iterations
        max_measurements : int, optional
            Maximum total measurements
        stopping_criterion : callable, optional
            Function(measurements, posterior) -> bool
        initial_measurements : list, optional
            Initial measurement results
        
        Returns
        -------
        dict
            Final results including posterior and measurements
        """
        # Initialize with any provided measurements
        if initial_measurements:
            self.measurements = initial_measurements
        
        iteration = 0
        
        while True:
            # Check stopping conditions
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Stopping: max_iterations reached ({iteration})")
                break
            
            if max_measurements and len(self.measurements) >= max_measurements:
                logger.info(f"Stopping: max_measurements reached ({len(self.measurements)})")
                break
            
            if stopping_criterion and stopping_criterion(self.measurements, self.posterior_samples):
                logger.info("Stopping: criterion met")
                break
            
            # Run iteration
            logger.info(f"\n=== Iteration {iteration + 1} ===")
            self.run_iteration()
            iteration += 1
        
        return {
            'n_measurements': len(self.measurements),
            'n_iterations': iteration,
            'measurements': self.measurements,
            'posterior_samples': self.posterior_samples,
            'mcts_stats': self.mcts.stats
        }
