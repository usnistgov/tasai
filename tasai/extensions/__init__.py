"""
TAS-AI Extensions

Advanced features for autonomous spin wave spectroscopy:

1. MCTS Planner: Monte Carlo Tree Search for batch measurement planning
2. GNN Hypothesis Generator: Graph Neural Network-based Hamiltonian proposal

These extensions enable:
- Improved efficiency through multi-step planning (vs greedy one-step-ahead)
- Closed-loop discovery by proposing candidate models from crystal structures
"""

from .mcts_planner import (
    MCTSPlanner,
    MCTSConfig,
    MCTSNode,
    MeasurementPoint,
    OptimisticMCTSPipeline,
    compare_mcts_vs_greedy
)

from .gnn_hypothesis import (
    GNNHypothesisGenerator,
    CandidateHamiltonian,
    MagneticSite,
    ExchangePathway,
    CrystalGraphBuilder,
    candidates_to_tasai_models,
    get_priors_from_candidates
)
from .goodenough_kanamori import (
    GoodenoughKanamoriAnalyzer,
    ExchangePath,
)

__all__ = [
    # MCTS
    'MCTSPlanner',
    'MCTSConfig', 
    'MCTSNode',
    'MeasurementPoint',
    'OptimisticMCTSPipeline',
    'compare_mcts_vs_greedy',
    
    # GNN Hypothesis
    'GNNHypothesisGenerator',
    'CandidateHamiltonian',
    'MagneticSite',
    'ExchangePathway',
    'CrystalGraphBuilder',
    'candidates_to_tasai_models',
    'get_priors_from_candidates',
    'GoodenoughKanamoriAnalyzer',
    'ExchangePath',
]
