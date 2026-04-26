"""
Entropy Estimation for Active Learning

This module provides entropy estimation functions for computing information
gain in Bayesian experimental design. Based on the methods used in AutoREFL.

Key functions:
- differential_entropy: Estimate entropy of continuous distribution from samples
- kl_divergence: Estimate KL divergence between distributions
- mutual_information: Estimate mutual information

References:
- Treece et al., J. Appl. Cryst. 52, 47-59 (2019) - Information theory framework
- Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192–1204 (2024) - AutoREFL

Methods implemented:
- K-nearest neighbor (KNN) entropy estimator (Kozachenko-Leonenko)
- Kernel density estimation (KDE) based entropy
- Binned entropy for 1D distributions
"""

import numpy as np
from typing import Optional, Union, List, Tuple
from scipy import stats
from scipy.special import digamma
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Core Entropy Estimators
# =============================================================================

def differential_entropy_knn(samples: np.ndarray, 
                              k: int = None,
                              normalize: bool = False) -> float:
    """
    Estimate differential entropy using k-nearest neighbor method.
    
    Uses the Kozachenko-Leonenko estimator, which is consistent and
    works well for multivariate distributions.
    
    H(X) ≈ ψ(N) - ψ(k) + d*log(2) + (d/N) * Σ log(ρ_i)
    
    where ρ_i is the distance to the k-th nearest neighbor of sample i.
    
    Parameters
    ----------
    samples : np.ndarray
        Samples from the distribution, shape (n_samples, n_dims)
    k : int, optional
        Number of neighbors. Default: max(1, n_samples // 50)
    normalize : bool
        If True, normalize samples to unit variance before estimation
    
    Returns
    -------
    float
        Estimated differential entropy in nats
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T
    
    n, d = samples.shape
    
    if n < 5:
        logger.warning(f"Very few samples ({n}) for entropy estimation")
        return 0.0
    
    # Default k
    if k is None:
        k = max(1, min(n // 50, 10))
    k = min(k, n - 1)
    
    # Normalize if requested
    if normalize:
        std = np.std(samples, axis=0)
        std[std < 1e-10] = 1.0
        samples = samples / std
        # Add log(prod(std)) to correct for normalization
        correction = np.sum(np.log(std[std > 1e-10]))
    else:
        correction = 0.0
    
    # Use KDTree for efficient neighbor search
    from scipy.spatial import KDTree
    tree = KDTree(samples)
    
    # Find distance to k-th nearest neighbor (index k because point is its own 0-th neighbor)
    distances, _ = tree.query(samples, k=k+1)
    rho = distances[:, -1]  # Distance to k-th neighbor
    
    # Avoid log(0)
    rho = np.maximum(rho, 1e-10)
    
    # Kozachenko-Leonenko estimator
    # H = ψ(N) - ψ(k) + log(V_d) + (d/N) * Σ log(ρ_i)
    # where V_d = π^(d/2) / Γ(d/2 + 1) is volume of unit d-ball
    
    # log(V_d) = (d/2)*log(π) - log(Γ(d/2 + 1))
    from scipy.special import gammaln
    log_V_d = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    
    entropy = (digamma(n) - digamma(k) + log_V_d + 
               (d / n) * np.sum(np.log(rho * 2)))  # *2 for diameter vs radius
    
    return entropy + correction


def differential_entropy_kde(samples: np.ndarray,
                              bandwidth: str = 'scott') -> float:
    """
    Estimate differential entropy using kernel density estimation.
    
    Better for low-dimensional distributions (d ≤ 3), more expensive
    than KNN for high dimensions.
    
    H(X) = -E[log p(X)] ≈ -(1/N) Σ log p(x_i)
    
    Parameters
    ----------
    samples : np.ndarray
        Samples from the distribution, shape (n_samples,) or (n_samples, n_dims)
    bandwidth : str
        Bandwidth selection method: 'scott' or 'silverman'
    
    Returns
    -------
    float
        Estimated differential entropy in nats
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T
    
    n, d = samples.shape
    
    if n < 5:
        return 0.0
    
    if d > 5:
        logger.warning(f"KDE entropy estimation may be slow for d={d} dimensions")
    
    try:
        # Fit KDE
        kde = stats.gaussian_kde(samples.T, bw_method=bandwidth)
        
        # Evaluate log probability at samples
        log_probs = kde.logpdf(samples.T)
        
        # Monte Carlo estimate of entropy
        entropy = -np.mean(log_probs)
        
    except np.linalg.LinAlgError:
        # Singular covariance, fall back to KNN
        logger.warning("KDE failed (singular covariance), using KNN")
        entropy = differential_entropy_knn(samples)
    
    return entropy


def differential_entropy_1d(samples: np.ndarray,
                             method: str = 'kde') -> float:
    """
    Estimate entropy for 1D distribution.
    
    Parameters
    ----------
    samples : np.ndarray
        1D samples
    method : str
        'kde' for kernel density, 'binned' for histogram
    
    Returns
    -------
    float
        Estimated entropy in nats
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)
    
    if n < 5:
        return 0.0
    
    if method == 'kde':
        kde = stats.gaussian_kde(samples)
        log_probs = kde.logpdf(samples)
        return -np.mean(log_probs)
    
    elif method == 'binned':
        # Use Freedman-Diaconis rule for bin width
        iqr = np.percentile(samples, 75) - np.percentile(samples, 25)
        bin_width = 2 * iqr / (n ** (1/3))
        
        if bin_width < 1e-10:
            bin_width = (np.max(samples) - np.min(samples)) / 20
        
        if bin_width < 1e-10:
            return 0.0
        
        n_bins = int(np.ceil((np.max(samples) - np.min(samples)) / bin_width))
        n_bins = max(5, min(n_bins, n // 5))
        
        counts, edges = np.histogram(samples, bins=n_bins)
        probs = counts / n
        probs = probs[probs > 0]
        
        # Discrete entropy + correction for continuous
        entropy = -np.sum(probs * np.log(probs)) + np.log(bin_width)
        return entropy
    
    else:
        raise ValueError(f"Unknown method: {method}")


def differential_entropy(samples: np.ndarray,
                          method: str = 'auto',
                          **kwargs) -> float:
    """
    Estimate differential entropy of samples.
    
    Parameters
    ----------
    samples : np.ndarray
        Samples from distribution, shape (n_samples,) or (n_samples, n_dims)
    method : str
        'knn', 'kde', 'binned', or 'auto'
    **kwargs
        Additional arguments passed to specific method
    
    Returns
    -------
    float
        Estimated differential entropy in nats
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T
    
    n, d = samples.shape
    
    if method == 'auto':
        if d == 1:
            method = 'kde'
        elif d <= 3 and n > 100:
            method = 'kde'
        else:
            method = 'knn'
    
    if method == 'knn':
        return differential_entropy_knn(samples, **kwargs)
    elif method == 'kde':
        return differential_entropy_kde(samples, **kwargs)
    elif method == 'binned':
        if d > 1:
            raise ValueError("Binned method only works for 1D")
        return differential_entropy_1d(samples.flatten(), method='binned')
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Information Gain and Related Quantities
# =============================================================================

def entropy_of_marginals(samples: np.ndarray,
                          indices: List[int] = None,
                          method: str = 'auto') -> float:
    """
    Compute sum of marginal entropies for selected parameters.
    
    H(X_1) + H(X_2) + ... + H(X_k)
    
    This is an upper bound on joint entropy: H(X) ≤ Σ H(X_i)
    
    Parameters
    ----------
    samples : np.ndarray
        Samples, shape (n_samples, n_dims)
    indices : list of int, optional
        Which dimensions to include. Default: all
    method : str
        Entropy estimation method
    
    Returns
    -------
    float
        Sum of marginal entropies
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T
    
    if indices is None:
        indices = list(range(samples.shape[1]))
    
    total = 0.0
    for i in indices:
        total += differential_entropy_1d(samples[:, i])
    
    return total


def joint_entropy(samples: np.ndarray,
                   indices: List[int] = None,
                   method: str = 'auto') -> float:
    """
    Compute joint entropy of selected parameters.
    
    H(X_1, X_2, ..., X_k)
    
    Parameters
    ----------
    samples : np.ndarray
        Samples, shape (n_samples, n_dims)
    indices : list of int, optional
        Which dimensions to include. Default: all
    method : str
        Entropy estimation method
    
    Returns
    -------
    float
        Joint entropy
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T
    
    if indices is None:
        subset = samples
    else:
        subset = samples[:, indices]
    
    return differential_entropy(subset, method=method)


def conditional_entropy(samples: np.ndarray,
                         target_indices: List[int],
                         given_indices: List[int],
                         method: str = 'auto') -> float:
    """
    Estimate conditional entropy H(X|Y).
    
    H(X|Y) = H(X,Y) - H(Y)
    
    Parameters
    ----------
    samples : np.ndarray
        Samples, shape (n_samples, n_dims)
    target_indices : list of int
        Indices of X variables
    given_indices : list of int
        Indices of Y variables (conditioning)
    method : str
        Entropy estimation method
    
    Returns
    -------
    float
        Conditional entropy H(X|Y)
    """
    all_indices = list(target_indices) + list(given_indices)
    
    H_XY = joint_entropy(samples, all_indices, method)
    H_Y = joint_entropy(samples, given_indices, method)
    
    return H_XY - H_Y


def mutual_information(samples: np.ndarray,
                        indices1: List[int],
                        indices2: List[int],
                        method: str = 'auto') -> float:
    """
    Estimate mutual information I(X;Y).
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Parameters
    ----------
    samples : np.ndarray
        Samples, shape (n_samples, n_dims)
    indices1 : list of int
        Indices of X variables
    indices2 : list of int
        Indices of Y variables
    method : str
        Entropy estimation method
    
    Returns
    -------
    float
        Mutual information I(X;Y)
    """
    H_X = joint_entropy(samples, indices1, method)
    H_Y = joint_entropy(samples, indices2, method)
    H_XY = joint_entropy(samples, list(indices1) + list(indices2), method)
    
    return H_X + H_Y - H_XY


def kl_divergence_samples(samples_p: np.ndarray,
                           samples_q: np.ndarray,
                           k: int = None) -> float:
    """
    Estimate KL divergence D_KL(P||Q) from samples.
    
    Uses the k-NN based estimator.
    
    Parameters
    ----------
    samples_p : np.ndarray
        Samples from P, shape (n_p, d)
    samples_q : np.ndarray
        Samples from Q, shape (n_q, d)
    k : int, optional
        Number of neighbors
    
    Returns
    -------
    float
        Estimated KL divergence
    """
    from scipy.spatial import KDTree
    
    samples_p = np.atleast_2d(samples_p)
    samples_q = np.atleast_2d(samples_q)
    
    if samples_p.shape[0] == 1:
        samples_p = samples_p.T
    if samples_q.shape[0] == 1:
        samples_q = samples_q.T
    
    n_p, d = samples_p.shape
    n_q = samples_q.shape[0]
    
    if k is None:
        k = max(1, min(n_p // 50, n_q // 50, 5))
    
    # Build trees
    tree_p = KDTree(samples_p)
    tree_q = KDTree(samples_q)
    
    # Distance to k-th neighbor in P and Q for each sample from P
    dist_p, _ = tree_p.query(samples_p, k=k+1)  # +1 for self
    rho = dist_p[:, -1]
    
    dist_q, _ = tree_q.query(samples_p, k=k)
    nu = dist_q[:, -1]
    
    # Avoid log(0)
    rho = np.maximum(rho, 1e-10)
    nu = np.maximum(nu, 1e-10)
    
    # KL estimator
    kl = (d / n_p) * np.sum(np.log(nu / rho)) + np.log(n_q / (n_p - 1))
    
    return max(0.0, kl)


# =============================================================================
# Information Gain for Active Learning
# =============================================================================

def expected_information_gain(prior_samples: np.ndarray,
                               likelihood_weights: np.ndarray,
                               poi_indices: List[int] = None,
                               method: str = 'auto') -> float:
    """
    Estimate expected information gain from a measurement.
    
    IG = H(prior) - E[H(posterior)]
    
    Uses importance sampling: posterior ~ prior * likelihood
    
    Parameters
    ----------
    prior_samples : np.ndarray
        Samples from prior, shape (n_samples, n_params)
    likelihood_weights : np.ndarray
        Likelihood weights for each sample (will be normalized)
    poi_indices : list of int, optional
        Parameters of interest. Default: all
    method : str
        Entropy estimation method
    
    Returns
    -------
    float
        Expected information gain in nats
    """
    n_samples = len(prior_samples)
    
    # Normalize weights
    weights = np.asarray(likelihood_weights)
    weights = weights / np.sum(weights)
    
    # Prior entropy
    if poi_indices is not None:
        prior_subset = prior_samples[:, poi_indices]
    else:
        prior_subset = prior_samples
    
    H_prior = differential_entropy(prior_subset, method=method)
    
    # Posterior entropy via resampling
    # Effective sample size check
    ess = 1.0 / np.sum(weights ** 2)
    
    if ess < 10:
        logger.warning(f"Low effective sample size: {ess:.1f}")
    
    # Resample according to weights
    n_resample = min(n_samples, int(ess * 10))
    indices = np.random.choice(n_samples, size=n_resample, p=weights, replace=True)
    posterior_samples = prior_samples[indices]
    
    if poi_indices is not None:
        posterior_subset = posterior_samples[:, poi_indices]
    else:
        posterior_subset = posterior_samples
    
    H_posterior = differential_entropy(posterior_subset, method=method)
    
    return H_prior - H_posterior


def information_rate(entropy_reduction: float,
                      count_time: float,
                      move_time: float,
                      eta: float = 1.0) -> float:
    """
    Compute information acquisition rate (AutoREFL's figure of merit).
    
    Rate = (ΔH)^η / (t_count + t_move)
    
    Parameters
    ----------
    entropy_reduction : float
        Expected entropy reduction ΔH
    count_time : float
        Counting time in seconds
    move_time : float
        Movement time in seconds
    eta : float
        Aggressiveness parameter (0 < η ≤ 1)
        η = 1: maximize information rate
        η < 1: favor faster measurements
    
    Returns
    -------
    float
        Information rate
    """
    if entropy_reduction <= 0:
        return 0.0
    
    total_time = count_time + move_time
    if total_time <= 0:
        return 0.0
    
    return (entropy_reduction ** eta) / total_time


# =============================================================================
# Utility Functions
# =============================================================================

def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size for weighted samples.
    
    ESS = (Σ w_i)² / Σ w_i²
    
    Parameters
    ----------
    weights : np.ndarray
        Importance weights (unnormalized OK)
    
    Returns
    -------
    float
        Effective sample size
    """
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights ** 2)


def resample_posterior(prior_samples: np.ndarray,
                        log_likelihood: np.ndarray,
                        n_resample: int = None) -> np.ndarray:
    """
    Resample from posterior using importance weights.
    
    Parameters
    ----------
    prior_samples : np.ndarray
        Samples from prior, shape (n, d)
    log_likelihood : np.ndarray
        Log likelihood for each sample, shape (n,)
    n_resample : int, optional
        Number of samples to draw. Default: same as input
    
    Returns
    -------
    np.ndarray
        Resampled posterior samples
    """
    n = len(prior_samples)
    if n_resample is None:
        n_resample = n
    
    # Convert to weights (with numerical stability)
    log_weights = log_likelihood - np.max(log_likelihood)
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    # Check ESS
    ess = effective_sample_size(weights)
    if ess < n / 10:
        logger.warning(f"Low ESS: {ess:.1f} / {n}")
    
    # Resample
    indices = np.random.choice(n, size=n_resample, p=weights, replace=True)
    return prior_samples[indices]


def gaussian_log_likelihood(predicted: np.ndarray,
                             observed: float,
                             uncertainty: float) -> np.ndarray:
    """
    Compute Gaussian log likelihood for array of predictions.
    
    Parameters
    ----------
    predicted : np.ndarray
        Predicted values for each parameter set
    observed : float
        Observed value
    uncertainty : float
        Measurement uncertainty (standard deviation)
    
    Returns
    -------
    np.ndarray
        Log likelihood for each prediction
    """
    return -0.5 * ((predicted - observed) / uncertainty) ** 2
