"""
MCMC Inference for Physics Models

Provides MCMC sampling for posterior inference using:
- BUMPS/DREAM (primary - used by AutoREFL/AutoNSE)
- emcee (alternative)
- Simple Metropolis-Hastings (fallback)

BUMPS with the DREAM algorithm (Differential Evolution Adaptive Metropolis)
is the standard for NIST neutron scattering autonomous systems.

Reference: https://github.com/bumps/bumps
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class BumpsProblem:
    """
    Wrapper to make a PhysicsModel compatible with BUMPS.
    
    BUMPS expects a "fitness" object with:
    - parameters: list of bumps.Parameter
    - nllf(): negative log likelihood function
    - numpoints(): number of data points
    """
    
    def __init__(self, 
                 model: 'PhysicsModel',
                 h: np.ndarray, k: np.ndarray, l: np.ndarray, E: np.ndarray,
                 I: np.ndarray, sigma: np.ndarray):
        self.model = model
        self.h = h
        self.k = k
        self.l = l
        self.E = E
        self.I = I
        self.sigma = sigma
        
        # Create BUMPS parameters
        try:
            from bumps.parameter import Parameter as BumpsParameter
            
            self._bumps_params = []
            for param in model.free_parameters:
                bp = BumpsParameter(
                    value=param.value,
                    bounds=param.bounds,
                    name=param.name
                )
                self._bumps_params.append(bp)
        except ImportError:
            self._bumps_params = None
    
    @property
    def parameters(self):
        """Return BUMPS parameters."""
        return self._bumps_params
    
    def numpoints(self):
        """Number of data points."""
        return len(self.I)
    
    def nllf(self):
        """Negative log likelihood (what BUMPS minimizes)."""
        # Sync parameter values from BUMPS to model
        for bp, mp in zip(self._bumps_params, self.model.free_parameters):
            mp.value = bp.value
        
        # Compute negative log likelihood
        log_like = self.model.log_likelihood(
            self.h, self.k, self.l, self.E, self.I, self.sigma
        )
        
        return -log_like
    
    def residuals(self):
        """Residuals for least-squares fitting."""
        I_pred = self.model.compute_intensity_array(self.h, self.k, self.l, self.E)
        return (self.I - I_pred) / self.sigma


class MCMCRunner:
    """
    MCMC runner for physics model parameter inference.
    
    Primary backend is BUMPS/DREAM (as used by AutoREFL).
    Falls back to emcee or simple Metropolis if BUMPS unavailable.
    
    Parallelization (emcee backend):
    - parallel=True: Use all CPU cores (recommended for M1 Mac)
    - parallel=4: Use 4 processes
    - parallel='threads': Use threads (avoids pickle issues)
    
    Typical timing on M1 Mac (8 cores):
    - Single-threaded: ~1.5-3 sec for 500+500 steps
    - parallel=True: ~0.3-0.5 sec (4-6x speedup)
    """
    
    def __init__(self,
                 model: 'PhysicsModel',
                 burn: int = 500,
                 steps: int = 500,
                 pop: int = 8,
                 backend: str = 'auto',
                 parallel: bool = False):
        """
        Initialize MCMC runner.
        
        Parameters
        ----------
        model : PhysicsModel
            Physics model with parameters to infer
        burn : int
            Burn-in steps (DREAM: iterations before sampling)
        steps : int
            Production steps (DREAM: iterations for sampling)
        pop : int
            Population size multiplier (DREAM uses pop * n_params chains)
            This matches AutoREFL's --pop parameter
        backend : str
            'bumps', 'emcee', 'metropolis', or 'auto'
        parallel : bool, int, or str
            Parallelization for emcee backend:
            - False: Single-threaded (default)
            - True: Use all CPU cores
            - int: Use specified number of processes
            - 'threads': Use ThreadPoolExecutor
        """
        self.model = model
        self.burn = burn
        self.steps = steps
        self.pop = pop  # DREAM population multiplier
        self.backend = backend
        self.parallel = parallel
        
        # For emcee compatibility (emcee uses fixed number of walkers)
        # Default: 2 * n_params or pop * n_params
        self._walkers = None
        
        # Data storage
        self.h_data: Optional[np.ndarray] = None
        self.k_data: Optional[np.ndarray] = None
        self.l_data: Optional[np.ndarray] = None
        self.E_data: Optional[np.ndarray] = None
        self.I_data: Optional[np.ndarray] = None
        self.sigma_data: Optional[np.ndarray] = None
        
        # Results
        self.chain: Optional[np.ndarray] = None
        self.log_prob: Optional[np.ndarray] = None
        self.dream_state = None  # Store DREAM state for analysis
    
    @property
    def walkers(self) -> int:
        """Number of walkers for emcee (defaults to pop * n_params)."""
        if self._walkers is not None:
            return self._walkers
        return max(self.pop * self.model.n_free, 16)
    
    @walkers.setter
    def walkers(self, value: int):
        self._walkers = value
    
    def set_data(self,
                 h: np.ndarray, k: np.ndarray, l: np.ndarray, E: np.ndarray,
                 I: np.ndarray, sigma: np.ndarray):
        """Set observed data for inference."""
        self.h_data = np.asarray(h)
        self.k_data = np.asarray(k)
        self.l_data = np.asarray(l)
        self.E_data = np.asarray(E)
        self.I_data = np.asarray(I)
        self.sigma_data = np.asarray(sigma)
    
    def log_posterior(self, params: np.ndarray) -> float:
        """Compute log posterior for parameter vector."""
        # Set model parameters
        self.model.set_free_values(params)
        
        # Check prior bounds
        log_prior = self.model.log_prior()
        if not np.isfinite(log_prior):
            return -np.inf
        
        # Compute likelihood
        log_like = self.model.log_likelihood(
            self.h_data, self.k_data, self.l_data, self.E_data,
            self.I_data, self.sigma_data
        )
        
        return log_prior + log_like
    
    def run(self,
            h: np.ndarray = None, k: np.ndarray = None,
            l: np.ndarray = None, E: np.ndarray = None,
            I: np.ndarray = None, sigma: np.ndarray = None,
            initial_state: np.ndarray = None) -> np.ndarray:
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        h, k, l, E, I, sigma : np.ndarray
            Observed data (if not already set)
        initial_state : np.ndarray, optional
            Initial chain state to resume from
        
        Returns
        -------
        np.ndarray
            Posterior samples, shape (n_samples, n_params)
        """
        if h is not None:
            self.set_data(h, k, l, E, I, sigma)
        
        if self.h_data is None:
            raise ValueError("No data set for inference")
        
        # Select backend
        backend = self._select_backend()
        
        logger.info(f"Running MCMC with {backend} backend")
        logger.info(f"  Burn: {self.burn}, Steps: {self.steps}, Pop: {self.pop}")
        logger.info(f"  Free parameters: {self.model.n_free}")
        
        if backend == 'bumps':
            self.chain = self._run_bumps(initial_state)
        elif backend == 'emcee':
            self.chain = self._run_emcee(initial_state)
        else:
            self.chain = self._run_metropolis(initial_state)
        
        logger.info(f"MCMC complete: {len(self.chain)} samples")
        
        return self.chain
    
    def _select_backend(self) -> str:
        """Select MCMC backend based on availability."""
        if self.backend != 'auto':
            return self.backend
        
        # Try bumps first (preferred, used by AutoREFL)
        try:
            import bumps.dream
            return 'bumps'
        except ImportError:
            pass
        
        # Try emcee
        try:
            import emcee
            return 'emcee'
        except ImportError:
            pass
        
        # Fallback to metropolis
        return 'metropolis'
    
    def _run_bumps(self, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Run MCMC using BUMPS/DREAM algorithm.
        
        DREAM = Differential Evolution Adaptive Metropolis.
        This is the standard MCMC algorithm for NIST autonomous systems.
        """
        try:
            from bumps.dream.core import run_dream
            from bumps.dream.model import MCMCModel
            from bumps.dream.state import load_state
            from bumps.fitproblem import FitProblem
            from bumps.parameter import Parameter as BumpsParameter
        except ImportError as e:
            logger.warning(f"BUMPS import failed: {e}, falling back to emcee")
            return self._run_emcee(initial_state)
        
        # Create BUMPS-compatible problem
        problem = BumpsProblem(
            self.model,
            self.h_data, self.k_data, self.l_data, self.E_data,
            self.I_data, self.sigma_data
        )
        
        # Wrap in FitProblem
        fit_problem = FitProblem(problem)
        
        # DREAM parameters (matching AutoREFL defaults)
        n_params = self.model.n_free
        n_chains = self.pop * n_params  # Population size
        
        # Create DREAM model
        dream_model = MCMCModel(fit_problem)
        
        # Initialize population
        if initial_state is not None:
            # Use provided initial state
            population = initial_state[-n_chains:, :]
        else:
            # Initialize from prior
            population = self.model.sample_prior(n_chains)
        
        # Set up temporary directory for DREAM state
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, 'dream_state')
            
            # Set up parallel mapper if requested
            mapper = None
            if self.parallel:
                try:
                    from bumps.mapper import MPMapper, can_pickle
                    import multiprocessing
                    
                    # Determine number of workers
                    if isinstance(self.parallel, int):
                        n_workers = self.parallel
                    else:
                        n_workers = multiprocessing.cpu_count()
                    
                    # Check if problem can be pickled (required for multiprocessing)
                    if can_pickle(fit_problem):
                        mapper = MPMapper.start_mapper(fit_problem, None, cpus=n_workers)
                        logger.info(f"BUMPS using {n_workers} parallel workers")
                    else:
                        logger.warning("Problem cannot be pickled, running serial")
                except Exception as e:
                    logger.warning(f"Parallel mapper setup failed: {e}, running serial")
            
            # Run DREAM
            logger.info(f"Running DREAM with {n_chains} chains...")
            
            state = run_dream(
                dream_model,
                population=population,
                burn=self.burn,
                steps=self.steps,
                thin=1,
                # Additional DREAM options
                DE_noise=1e-6,
                outlier_test='IQR',
                mapper=mapper,
            )
            
            # Store state for later analysis
            self.dream_state = state
            
            # Cleanup parallel mapper
            if mapper is not None:
                try:
                    mapper.stop_mapper()
                except:
                    pass
            
            # Extract chain (all samples after burn-in)
            # DREAM returns shape (n_steps, n_chains, n_params)
            draws = state.draw()
            
            # Flatten to (n_samples, n_params)
            chain = draws.reshape(-1, n_params)
            
            # Also extract log probabilities
            self.log_prob = -state.logp().flatten()  # DREAM stores -logp
        
        return chain
    
    def _run_bumps_simple(self, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Simplified BUMPS/DREAM run using bumps.fitters.
        
        This is an alternative approach using the higher-level bumps API.
        """
        try:
            from bumps.fitters import fit, DreamFit
            from bumps.fitproblem import FitProblem
        except ImportError:
            logger.warning("BUMPS not available, falling back to emcee")
            return self._run_emcee(initial_state)
        
        # Create problem
        problem = BumpsProblem(
            self.model,
            self.h_data, self.k_data, self.l_data, self.E_data,
            self.I_data, self.sigma_data
        )
        
        fit_problem = FitProblem(problem)
        
        # Run DREAM fit
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fit(
                fit_problem,
                method='dream',
                burn=self.burn,
                steps=self.steps,
                pop=self.pop,
                store=tmpdir
            )
            
            # Load state to get chain
            from bumps.dream.state import load_state
            state = load_state(os.path.join(tmpdir, 'model'))
            
            draws = state.draw()
            chain = draws.reshape(-1, self.model.n_free)
        
        return chain
    
    def _run_emcee(self, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Run MCMC using emcee with optional parallelization.
        
        Parallelization options (set via self.parallel):
        - None or False: Single-threaded (default)
        - True or 'auto': Use all available CPU cores
        - int: Use specified number of processes
        - 'threads': Use ThreadPoolExecutor (good for I/O bound)
        
        On M1 Mac with 8 cores:
        - Single-threaded: ~1.5-3 sec for 500+500 steps
        - 8 cores parallel: ~0.3-0.5 sec (4-6x speedup)
        """
        try:
            import emcee
        except ImportError:
            logger.warning("emcee not available, falling back to metropolis")
            return self._run_metropolis(initial_state)
        
        n_dim = self.model.n_free
        
        # Initialize walkers
        if initial_state is not None:
            p0 = initial_state[-self.walkers:, :]
            p0 = p0 + 1e-4 * np.random.randn(self.walkers, n_dim)
        else:
            p0 = self.model.sample_prior(self.walkers)
        
        # Set up parallelization
        pool = None
        n_workers = None
        
        if hasattr(self, 'parallel') and self.parallel:
            import multiprocessing
            import os
            
            if self.parallel == 'threads':
                # Thread-based parallelism (avoids pickle issues)
                from concurrent.futures import ThreadPoolExecutor
                n_workers = os.cpu_count() or 4
                pool = ThreadPoolExecutor(max_workers=n_workers)
                logger.info(f"Using {n_workers} threads for MCMC")
            else:
                # Process-based parallelism (true parallel on M1)
                if isinstance(self.parallel, int):
                    n_workers = self.parallel
                else:
                    n_workers = os.cpu_count() or 4
                
                # Use 'fork' on macOS for better performance
                # Note: 'spawn' is safer but slower
                try:
                    multiprocessing.set_start_method('fork', force=True)
                except RuntimeError:
                    pass  # Already set
                
                pool = multiprocessing.Pool(n_workers)
                logger.info(f"Using {n_workers} processes for MCMC")
        
        try:
            # Create sampler with optional pool
            sampler = emcee.EnsembleSampler(
                self.walkers, n_dim, self.log_posterior,
                pool=pool
            )
            
            # Burn-in
            logger.info(f"Running burn-in ({self.burn} steps)...")
            state = sampler.run_mcmc(p0, self.burn, progress=False)
            sampler.reset()
            
            # Production
            logger.info(f"Running production ({self.steps} steps)...")
            sampler.run_mcmc(state, self.steps, progress=False)
            
        finally:
            # Clean up pool
            if pool is not None:
                if hasattr(pool, 'shutdown'):
                    pool.shutdown(wait=True)
                elif hasattr(pool, 'close'):
                    pool.close()
                    pool.join()
        
        # Flatten chains
        self.chain = sampler.get_chain(flat=True)
        self.log_prob = sampler.get_log_prob(flat=True)
        
        return self.chain
    
    def _run_metropolis(self, initial_state: np.ndarray = None) -> np.ndarray:
        """Simple Metropolis-Hastings MCMC."""
        n_dim = self.model.n_free
        n_samples = self.walkers * self.steps
        
        # Initialize
        if initial_state is not None:
            current = initial_state[-1, :]
        else:
            current = self.model.sample_prior(1)[0]
        
        current_lp = self.log_posterior(current)
        
        # Proposal scale based on parameter bounds
        free_params = self.model.free_parameters
        scales = np.array([
            (p.bounds[1] - p.bounds[0]) * 0.1 for p in free_params
        ])
        
        # Run chain
        chain = np.zeros((n_samples + self.burn, n_dim))
        log_probs = np.zeros(n_samples + self.burn)
        
        n_accept = 0
        
        for i in range(n_samples + self.burn):
            # Propose
            proposal = current + scales * np.random.randn(n_dim)
            
            # Clip to bounds
            for j, param in enumerate(free_params):
                proposal[j] = np.clip(proposal[j], param.bounds[0], param.bounds[1])
            
            proposal_lp = self.log_posterior(proposal)
            
            # Accept/reject
            if np.log(np.random.random()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
                n_accept += 1
            
            chain[i] = current
            log_probs[i] = current_lp
            
            # Adapt proposal scale during burn-in
            if i < self.burn and i > 100 and i % 100 == 0:
                accept_rate = n_accept / (i + 1)
                if accept_rate < 0.2:
                    scales *= 0.8
                elif accept_rate > 0.5:
                    scales *= 1.2
        
        logger.info(f"Metropolis acceptance rate: {n_accept / (n_samples + self.burn):.2%}")
        
        # Discard burn-in
        self.chain = chain[self.burn:]
        self.log_prob = log_probs[self.burn:]
        
        return self.chain
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each parameter."""
        if self.chain is None:
            raise ValueError("No chain available - run MCMC first")
        
        summary = {}
        for i, param in enumerate(self.model.free_parameters):
            samples = self.chain[:, i]
            summary[param.name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'q16': np.percentile(samples, 16),
                'q84': np.percentile(samples, 84),
            }
        
        return summary
    
    def get_best_fit(self) -> Dict[str, float]:
        """Get maximum a posteriori (MAP) parameter values."""
        if self.chain is None or self.log_prob is None:
            raise ValueError("No chain available - run MCMC first")
        
        best_idx = np.argmax(self.log_prob)
        best_params = self.chain[best_idx]
        
        return {
            param.name: best_params[i] 
            for i, param in enumerate(self.model.free_parameters)
        }


def compute_bayes_factor(model1: 'PhysicsModel', 
                        model2: 'PhysicsModel',
                        h: np.ndarray, k: np.ndarray, l: np.ndarray, E: np.ndarray,
                        I: np.ndarray, sigma: np.ndarray,
                        n_samples: int = 1000) -> float:
    """
    Compute Bayes factor comparing two models.
    
    B₁₂ = P(data|model1) / P(data|model2)
    
    Uses simple harmonic mean estimator (not the most accurate but simple).
    
    Returns log(B₁₂): positive favors model1, negative favors model2.
    """
    def log_marginal_likelihood(model: 'PhysicsModel') -> float:
        """Estimate log marginal likelihood via harmonic mean."""
        # Run short MCMC
        runner = MCMCRunner(model, burn=200, steps=n_samples // 10, walkers=8)
        runner.set_data(h, k, l, E, I, sigma)
        chain = runner.run()
        
        # Compute log likelihoods
        log_likes = []
        for params in chain[::10]:  # Thin
            model.set_free_values(params)
            ll = model.log_likelihood(h, k, l, E, I, sigma)
            log_likes.append(ll)
        
        log_likes = np.array(log_likes)
        
        # Harmonic mean estimator (with numerical stabilization)
        max_ll = np.max(log_likes)
        log_ml = max_ll - np.log(np.mean(np.exp(max_ll - log_likes)))
        
        return log_ml
    
    log_ml1 = log_marginal_likelihood(model1)
    log_ml2 = log_marginal_likelihood(model2)
    
    return log_ml1 - log_ml2
