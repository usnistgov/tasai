"""
TAS-AI Inference Module

MCMC sampling for Bayesian parameter inference.
Primary backend: BUMPS/DREAM (standard for NIST autonomous systems)
"""

from .mcmc import (
    MCMCRunner,
    BumpsProblem,
    compute_bayes_factor,
)

__all__ = [
    'MCMCRunner',
    'BumpsProblem',
    'compute_bayes_factor',
]
