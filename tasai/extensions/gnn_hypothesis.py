"""
GNN-Assisted Hypothesis Generation for Magnetic Hamiltonians

This module uses Graph Neural Networks to propose candidate spin Hamiltonians
from crystal structures. It bridges the gap between structure and magnetic
properties, enabling fully autonomous characterization workflows.

The approach:
1. Encode crystal structure as a graph (atoms = nodes, bonds = edges)
2. Use pretrained GNN (CHGNet, M3GNet, or similar) to predict:
   - Magnetic moments per site
   - Total energy for different spin configurations
3. Extract exchange interactions via energy mapping
4. Generate candidate Hamiltonians with uncertainty estimates
5. Pass candidates to TAS-AI for experimental discrimination

This creates a closed loop:
   Structure → GNN → Candidates → TAS-AI → Validated J values → GNN training

References:
- Deng et al., Nat. Mach. Intell. 5, 1031 (2023) - CHGNet
- Chen et al., Nat. Comput. Sci. 2, 718 (2022) - M3GNet  
- Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018) - CGCNN
- MagNet: arXiv:2404.12406 (2024) - Equivariant NN for magnetic materials
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from .goodenough_kanamori import GoodenoughKanamoriAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class MagneticSite:
    """A magnetic ion site in the crystal."""
    index: int
    element: str
    position: np.ndarray  # Fractional coordinates
    magnetic_moment: float = 0.0
    spin: float = 0.5
    oxidation_state: int = 0


@dataclass
class ExchangePathway:
    """An exchange interaction pathway between two sites."""
    site1_idx: int
    site2_idx: int
    distance: float
    bond_type: str  # 'nn', 'nnn', 'inter_layer', etc.
    bridging_atoms: List[str] = field(default_factory=list)
    estimated_J: float = 0.0
    J_uncertainty: float = 0.0
    superexchange_angle: float = 0.0  # M-O-M angle for superexchange


@dataclass
class CandidateHamiltonian:
    """A candidate spin Hamiltonian."""
    name: str
    exchange_terms: Dict[str, float]  # {'J1': 5.0, 'J2': 0.5, 'D': 0.1}
    uncertainties: Dict[str, float]   # {'J1': 1.0, 'J2': 0.3, 'D': 0.05}
    prior_probability: float = 1.0
    description: str = ""
    magnetic_sites: List[MagneticSite] = field(default_factory=list)
    pathways: List[ExchangePathway] = field(default_factory=list)


class CrystalGraphBuilder:
    """
    Build graph representation of crystal structure for GNN input.
    
    Nodes: atoms with features (element, position, coordination)
    Edges: bonds with features (distance, bond type)
    """
    
    def __init__(self, 
                 cutoff: float = 6.0,
                 max_neighbors: int = 12):
        """
        Initialize graph builder.
        
        Parameters
        ----------
        cutoff : float
            Distance cutoff for edges (Angstroms)
        max_neighbors : int
            Maximum neighbors per atom
        """
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        # Element features (simplified - full version would use more)
        self.element_features = {
            'Fe': {'Z': 26, 'mass': 55.85, 'spin_default': 2.5, 'magnetic': True},
            'Co': {'Z': 27, 'mass': 58.93, 'spin_default': 1.5, 'magnetic': True},
            'Ni': {'Z': 28, 'mass': 58.69, 'spin_default': 1.0, 'magnetic': True},
            'Mn': {'Z': 25, 'mass': 54.94, 'spin_default': 2.5, 'magnetic': True},
            'Cr': {'Z': 24, 'mass': 52.00, 'spin_default': 1.5, 'magnetic': True},
            'Cu': {'Z': 29, 'mass': 63.55, 'spin_default': 0.5, 'magnetic': True},
            'Gd': {'Z': 64, 'mass': 157.25, 'spin_default': 3.5, 'magnetic': True},
            'Tb': {'Z': 65, 'mass': 158.93, 'spin_default': 3.0, 'magnetic': True},
            'Dy': {'Z': 66, 'mass': 162.50, 'spin_default': 2.5, 'magnetic': True},
            'O': {'Z': 8, 'mass': 16.00, 'spin_default': 0.0, 'magnetic': False},
            'S': {'Z': 16, 'mass': 32.07, 'spin_default': 0.0, 'magnetic': False},
            'Se': {'Z': 34, 'mass': 78.97, 'spin_default': 0.0, 'magnetic': False},
            'Cl': {'Z': 17, 'mass': 35.45, 'spin_default': 0.0, 'magnetic': False},
            'Br': {'Z': 35, 'mass': 79.90, 'spin_default': 0.0, 'magnetic': False},
            'I': {'Z': 53, 'mass': 126.90, 'spin_default': 0.0, 'magnetic': False},
        }
    
    def build_graph(self, structure: Any) -> Dict[str, Any]:
        """
        Build graph from structure.
        
        Parameters
        ----------
        structure : pymatgen Structure or dict
            Crystal structure
        
        Returns
        -------
        dict
            Graph with nodes, edges, and features
        """
        # Handle different structure formats
        if hasattr(structure, 'sites'):
            # pymatgen Structure
            return self._build_from_pymatgen(structure)
        elif isinstance(structure, dict):
            # Dictionary format
            return self._build_from_dict(structure)
        else:
            raise ValueError(f"Unknown structure type: {type(structure)}")
    
    def _build_from_dict(self, structure: Dict) -> Dict[str, Any]:
        """Build graph from dictionary structure."""
        lattice = np.array(structure.get('lattice', np.eye(3) * 5.0))
        species = structure.get('species', [])
        coords = np.array(structure.get('coords', []))
        
        n_atoms = len(species)
        
        # Node features
        node_features = []
        magnetic_mask = []
        
        for i, elem in enumerate(species):
            feat = self.element_features.get(elem, {
                'Z': 0, 'mass': 1.0, 'spin_default': 0.0, 'magnetic': False
            })
            node_features.append([
                feat['Z'],
                feat['mass'],
                feat['spin_default'],
                1.0 if feat['magnetic'] else 0.0
            ])
            magnetic_mask.append(feat['magnetic'])
        
        # Compute distances and edges
        edges = []
        edge_features = []
        
        # Convert fractional to Cartesian
        cart_coords = coords @ lattice
        
        for i in range(n_atoms):
            distances = []
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Handle periodic boundaries
                for da in [-1, 0, 1]:
                    for db in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            offset = np.array([da, db, dc]) @ lattice
                            dist = np.linalg.norm(cart_coords[j] + offset - cart_coords[i])
                            
                            if dist < self.cutoff:
                                distances.append((j, dist, offset))
            
            # Keep closest neighbors
            distances.sort(key=lambda x: x[1])
            
            for j, dist, offset in distances[:self.max_neighbors]:
                edges.append([i, j])
                edge_features.append([dist])
        
        return {
            'n_atoms': n_atoms,
            'species': species,
            'coords': coords,
            'lattice': lattice,
            'node_features': np.array(node_features),
            'edges': np.array(edges) if edges else np.zeros((0, 2), dtype=int),
            'edge_features': np.array(edge_features) if edge_features else np.zeros((0, 1)),
            'magnetic_mask': np.array(magnetic_mask)
        }
    
    def _build_from_pymatgen(self, structure) -> Dict[str, Any]:
        """Build graph from pymatgen Structure."""
        struct_dict = {
            'lattice': structure.lattice.matrix,
            'species': [str(site.specie) for site in structure.sites],
            'coords': structure.frac_coords
        }
        return self._build_from_dict(struct_dict)


class GNNHypothesisGenerator:
    """
    Generate candidate Hamiltonians using GNN predictions.
    
    This class can use various GNN backends:
    - CHGNet (pretrained, includes magnetic moments)
    - M3GNet (pretrained, energy predictions)
    - Custom PyTorch models
    
    If no GNN is available, falls back to rule-based heuristics.
    """
    
    def __init__(self,
                 gnn_backend: str = 'heuristic',
                 model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize hypothesis generator.
        
        Parameters
        ----------
        gnn_backend : str
            'chgnet', 'm3gnet', 'custom', or 'heuristic'
        model_path : str, optional
            Path to custom model weights
        device : str
            'cpu' or 'cuda'
        """
        self.gnn_backend = gnn_backend
        self.model_path = model_path
        self.device = device
        
        self.graph_builder = CrystalGraphBuilder()
        self.model = None
        self._last_gk_paths = []
        
        # Try to load GNN model
        self._load_model()
    
    def _load_model(self):
        """Load GNN model if available."""
        if self.gnn_backend == 'heuristic':
            logger.info("Using heuristic-based hypothesis generation (no GNN)")
            return
        
        try:
            if self.gnn_backend == 'chgnet':
                from chgnet.model import CHGNet
                self.model = CHGNet.load()
                logger.info("Loaded CHGNet model")
                
            elif self.gnn_backend == 'm3gnet':
                import matgl
                self.model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                logger.info("Loaded M3GNet model")
                
            elif self.gnn_backend == 'custom':
                if self.model_path:
                    import torch
                    self.model = torch.load(self.model_path, map_location=self.device)
                    logger.info(f"Loaded custom model from {self.model_path}")
                else:
                    logger.warning("No model path provided for custom backend")
                    self.gnn_backend = 'heuristic'
                    
        except ImportError as e:
            logger.warning(f"Could not load {self.gnn_backend}: {e}")
            logger.warning("Falling back to heuristic-based generation")
            self.gnn_backend = 'heuristic'
    
    def generate_candidates(self, 
                           structure: Any,
                           max_candidates: int = 5,
                           include_uncertainties: bool = True) -> List[CandidateHamiltonian]:
        """
        Generate candidate Hamiltonians from crystal structure.
        
        Parameters
        ----------
        structure : Structure or dict
            Crystal structure
        max_candidates : int
            Maximum number of candidates to generate
        include_uncertainties : bool
            Whether to estimate uncertainties
        
        Returns
        -------
        list of CandidateHamiltonian
            Ranked candidate Hamiltonians
        """
        # Build graph
        graph = self.graph_builder.build_graph(structure)
        
        # Identify magnetic sites
        magnetic_sites = self._identify_magnetic_sites(graph)
        
        if not magnetic_sites:
            logger.warning("No magnetic sites found in structure")
            return []
        
        # Identify exchange pathways
        pathways = self._identify_pathways(graph, magnetic_sites)
        
        # Estimate exchange interactions
        if self.gnn_backend == 'heuristic':
            J_estimates = self._estimate_J_heuristic(graph, pathways)
        else:
            J_estimates = self._estimate_J_gnn(graph, pathways)
        
        # Generate candidate Hamiltonians
        candidates = self._build_candidates(
            magnetic_sites, pathways, J_estimates, max_candidates
        )
        
        # Add uncertainties
        if include_uncertainties:
            self._add_uncertainties(candidates, J_estimates)
        
        # Rank by prior probability
        candidates.sort(key=lambda c: -c.prior_probability)
        
        return candidates[:max_candidates]
    
    def _identify_magnetic_sites(self, graph: Dict) -> List[MagneticSite]:
        """Identify magnetic sites in the structure."""
        sites = []
        
        for i, (elem, is_mag) in enumerate(zip(graph['species'], graph['magnetic_mask'])):
            if is_mag:
                feat = self.graph_builder.element_features.get(elem, {})
                sites.append(MagneticSite(
                    index=i,
                    element=elem,
                    position=graph['coords'][i],
                    magnetic_moment=feat.get('spin_default', 1.0) * 2,  # μ_B
                    spin=feat.get('spin_default', 0.5)
                ))
        
        return sites
    
    def _identify_pathways(self, 
                          graph: Dict, 
                          magnetic_sites: List[MagneticSite]) -> List[ExchangePathway]:
        """Identify exchange pathways between magnetic sites."""
        analyzer = GoodenoughKanamoriAnalyzer(graph)
        gk_paths = analyzer.find_exchange_paths(max_distance=self.graph_builder.cutoff)
        self._last_gk_paths = gk_paths

        ranked = analyzer.rank_paths(gk_paths)
        top_distances = sorted({round(path.distance, 3) for path in ranked})

        def bond_type_for(distance: float) -> str:
            for idx, ref_distance in enumerate(top_distances[:4]):
                if abs(distance - ref_distance) < 0.12:
                    return {0: 'nn', 1: 'nnn', 2: 'third_nn', 3: 'fourth_nn'}.get(idx, f'J{idx + 1}')
            if distance < 4.5:
                return 'nn'
            if distance < 6.5:
                return 'nnn'
            return 'long_range'

        pathways = []
        for path in ranked:
            pathways.append(ExchangePathway(
                site1_idx=path.site1,
                site2_idx=path.site2,
                distance=path.distance,
                bond_type=bond_type_for(path.distance),
                bridging_atoms=list(path.bridging_atoms),
                estimated_J=path.signed_strength,
                J_uncertainty=max(abs(path.signed_strength) * (1.0 - path.confidence), 0.05),
                superexchange_angle=path.bond_angle
            ))
        return pathways
    
    def _find_bridging_atoms(self, 
                            graph: Dict, 
                            idx1: int, 
                            idx2: int) -> List[str]:
        """Find atoms bridging two magnetic sites."""
        bridging = []
        cart_coords = graph['coords'] @ graph['lattice']
        
        midpoint = (cart_coords[idx1] + cart_coords[idx2]) / 2
        
        for i, elem in enumerate(graph['species']):
            if i in [idx1, idx2]:
                continue
            
            # Check if atom is between the two magnetic sites
            dist_to_mid = np.linalg.norm(cart_coords[i] - midpoint)
            dist_12 = np.linalg.norm(cart_coords[idx2] - cart_coords[idx1])
            
            if dist_to_mid < dist_12 / 2:
                bridging.append(elem)
        
        return bridging
    
    def _compute_superexchange_angle(self,
                                     graph: Dict,
                                     idx1: int,
                                     idx2: int,
                                     bridging: List[str]) -> float:
        """Compute M-O-M superexchange angle."""
        if not bridging:
            return 180.0  # Direct exchange
        
        cart_coords = graph['coords'] @ graph['lattice']
        pos1 = cart_coords[idx1]
        pos2 = cart_coords[idx2]
        
        # Find closest bridging atom
        best_angle = 180.0
        
        for i, elem in enumerate(graph['species']):
            if elem in bridging:
                pos_bridge = cart_coords[i]
                
                # Compute angle
                v1 = pos1 - pos_bridge
                v2 = pos2 - pos_bridge
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if abs(angle - 180) < abs(best_angle - 180):
                    best_angle = angle
        
        return best_angle
    
    def _estimate_J_heuristic(self, 
                              graph: Dict, 
                              pathways: List[ExchangePathway]) -> Dict[str, Tuple[float, float]]:
        """
        Estimate exchange interactions using heuristics.
        
        Based on:
        - Goodenough-Kanamori rules for superexchange
        - Distance dependence (J ~ 1/r^n)
        - Empirical values from literature
        """
        estimates = {}

        for term, bond_type in (('J1', 'nn'), ('J2', 'nnn')):
            matching = [p for p in pathways if p.bond_type == bond_type]
            if not matching:
                continue
            weights = np.array([1.0 / max(p.J_uncertainty, 0.05) ** 2 for p in matching], dtype=float)
            values = np.array([p.estimated_J for p in matching], dtype=float)
            estimate = float(np.average(values, weights=weights))
            scatter = float(np.sqrt(np.average((values - estimate) ** 2, weights=weights))) if len(values) > 1 else 0.0
            uncertainty = max(scatter, float(np.mean([p.J_uncertainty for p in matching])))
            estimates[term] = (estimate, uncertainty)

            for p in matching:
                p.estimated_J = estimate
                p.J_uncertainty = uncertainty

        # Single-ion anisotropy (heuristic based on ion type)
        magnetic_elements = set(graph['species'][i] for i in range(len(graph['species'])) 
                               if graph['magnetic_mask'][i])
        
        # Rare earths have large anisotropy
        if magnetic_elements & {'Gd', 'Tb', 'Dy', 'Ho', 'Er'}:
            D = 0.5  # meV, easy-axis
            D_err = 0.3
        elif magnetic_elements & {'Co', 'Fe'}:
            D = 0.1
            D_err = 0.1
        else:
            D = 0.01
            D_err = 0.05
        
        estimates['D'] = (D, D_err)
        
        return estimates
    
    def _estimate_J_gnn(self, 
                        graph: Dict, 
                        pathways: List[ExchangePathway]) -> Dict[str, Tuple[float, float]]:
        """
        Estimate exchange interactions using GNN energy mapping.
        
        Method:
        1. Predict energy for FM configuration (all spins up)
        2. Predict energy for various AFM configurations
        3. Extract J from energy differences
        """
        if self.model is None:
            logger.warning("No GNN model loaded, falling back to heuristic")
            return self._estimate_J_heuristic(graph, pathways)
        
        try:
            if self.gnn_backend == 'chgnet':
                return self._estimate_J_chgnet(graph, pathways)
            elif self.gnn_backend == 'm3gnet':
                return self._estimate_J_m3gnet(graph, pathways)
            else:
                return self._estimate_J_heuristic(graph, pathways)
        except Exception as e:
            logger.warning(f"GNN estimation failed: {e}, falling back to heuristic")
            return self._estimate_J_heuristic(graph, pathways)
    
    def _estimate_J_chgnet(self, 
                          graph: Dict, 
                          pathways: List[ExchangePathway]) -> Dict[str, Tuple[float, float]]:
        """Estimate J using CHGNet energy mapping."""
        from pymatgen.core import Structure, Lattice
        
        # Reconstruct pymatgen structure
        lattice = Lattice(graph['lattice'])
        structure = Structure(lattice, graph['species'], graph['coords'])
        
        # Get magnetic sites
        mag_indices = [i for i, m in enumerate(graph['magnetic_mask']) if m]
        n_mag = len(mag_indices)
        
        if n_mag == 0:
            return self._estimate_J_heuristic(graph, pathways)
        
        # Predict for FM configuration
        # CHGNet predicts magnetic moments, so we can check consistency
        prediction_fm = self.model.predict_structure(structure)
        E_fm = prediction_fm['e']
        magmoms_fm = prediction_fm.get('m', np.zeros(len(structure)))
        
        # For AFM, we'd need to constrain magnetic moments
        # This is a simplified version - full implementation would use
        # constrained DFT or spin-polarized calculations
        
        # Estimate from magnetic moments and typical J scaling
        avg_moment = np.mean(np.abs(magmoms_fm[mag_indices])) if mag_indices else 2.0
        S = avg_moment / 2  # Approximate spin
        
        # Use heuristic with GNN-informed spin value
        estimates = self._estimate_J_heuristic(graph, pathways)
        
        # Scale by actual predicted moment
        for key in estimates:
            val, err = estimates[key]
            estimates[key] = (val * (S / 1.0) ** 2, err * (S / 1.0) ** 2)
        
        return estimates
    
    def _estimate_J_m3gnet(self, 
                          graph: Dict, 
                          pathways: List[ExchangePathway]) -> Dict[str, Tuple[float, float]]:
        """Estimate J using M3GNet."""
        # M3GNet doesn't directly predict magnetic properties
        # Fall back to heuristic with energy-based refinement
        return self._estimate_J_heuristic(graph, pathways)
    
    def _build_candidates(self,
                         magnetic_sites: List[MagneticSite],
                         pathways: List[ExchangePathway],
                         J_estimates: Dict[str, Tuple[float, float]],
                         max_candidates: int) -> List[CandidateHamiltonian]:
        """Build candidate Hamiltonians from estimates."""
        candidates = []
        
        # Get estimates
        J1, J1_err = J_estimates.get('J1', (5.0, 2.5))
        J2, J2_err = J_estimates.get('J2', (0.0, 0.5))
        D, D_err = J_estimates.get('D', (0.0, 0.1))
        
        # Model 1: NN Heisenberg only
        candidates.append(CandidateHamiltonian(
            name='NN-Heisenberg',
            exchange_terms={'J1': J1},
            uncertainties={'J1': J1_err},
            prior_probability=0.4,
            description='Nearest-neighbor Heisenberg exchange only',
            magnetic_sites=magnetic_sites,
            pathways=[p for p in pathways if p.bond_type == 'nn']
        ))
        
        # Model 2: J1-J2 Heisenberg
        if J2 != 0:
            candidates.append(CandidateHamiltonian(
                name='J1-J2-Heisenberg',
                exchange_terms={'J1': J1, 'J2': J2},
                uncertainties={'J1': J1_err, 'J2': J2_err},
                prior_probability=0.3,
                description='NN + NNN Heisenberg exchange',
                magnetic_sites=magnetic_sites,
                pathways=pathways
            ))
        
        # Model 3: J1 + Anisotropy
        candidates.append(CandidateHamiltonian(
            name='J1-Anisotropy',
            exchange_terms={'J1': J1, 'D': D},
            uncertainties={'J1': J1_err, 'D': D_err},
            prior_probability=0.2,
            description='NN Heisenberg with single-ion anisotropy',
            magnetic_sites=magnetic_sites,
            pathways=[p for p in pathways if p.bond_type == 'nn']
        ))
        
        # Model 4: Full J1-J2-D
        if J2 != 0:
            candidates.append(CandidateHamiltonian(
                name='J1-J2-D',
                exchange_terms={'J1': J1, 'J2': J2, 'D': D},
                uncertainties={'J1': J1_err, 'J2': J2_err, 'D': D_err},
                prior_probability=0.1,
                description='Full model with J1, J2, and anisotropy',
                magnetic_sites=magnetic_sites,
                pathways=pathways
            ))
        
        # Normalize probabilities
        total_prob = sum(c.prior_probability for c in candidates)
        for c in candidates:
            c.prior_probability /= total_prob
        
        return candidates
    
    def _add_uncertainties(self,
                          candidates: List[CandidateHamiltonian],
                          J_estimates: Dict[str, Tuple[float, float]]):
        """Add uncertainty estimates to candidates."""
        for candidate in candidates:
            for term, val in candidate.exchange_terms.items():
                if term in J_estimates:
                    _, err = J_estimates[term]
                    candidate.uncertainties[term] = err


# =============================================================================
# Integration with TAS-AI
# =============================================================================

def candidates_to_tasai_models(
    candidates: List[CandidateHamiltonian],
    model_class: Any = None
) -> List[Any]:
    """
    Convert candidate Hamiltonians to TAS-AI physics models.
    
    Parameters
    ----------
    candidates : list of CandidateHamiltonian
        Candidates from GNN generator
    model_class : class, optional
        Physics model class to instantiate
    
    Returns
    -------
    list of PhysicsModel
        Models ready for TAS-AI
    """
    if model_class is None:
        # Use built-in square lattice FM model
        from ..sunny import SquareLatticeFM
        model_class = SquareLatticeFM
    
    models = []
    
    for candidate in candidates:
        params = candidate.exchange_terms.copy()
        
        # Map parameter names
        param_map = {
            'J1': 'J1',
            'J2': 'J2', 
            'D': 'D'
        }
        
        mapped_params = {
            param_map.get(k, k): v 
            for k, v in params.items()
        }
        
        try:
            model = model_class(**mapped_params)
            model.name = candidate.name
            model.prior_probability = candidate.prior_probability
            models.append(model)
        except Exception as e:
            logger.warning(f"Could not create model for {candidate.name}: {e}")
    
    return models


def get_priors_from_candidates(
    candidates: List[CandidateHamiltonian]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Extract parameter priors from GNN candidates.
    
    Returns priors in format suitable for MCMC:
    {
        'model_name': {
            'J1': (mean, std),
            'J2': (mean, std),
            ...
        }
    }
    """
    priors = {}
    
    for candidate in candidates:
        model_priors = {}
        
        for param, value in candidate.exchange_terms.items():
            uncertainty = candidate.uncertainties.get(param, abs(value) * 0.5)
            model_priors[param] = (value, uncertainty)
        
        priors[candidate.name] = model_priors
    
    return priors


# =============================================================================
# Example Usage
# =============================================================================

def demo_hypothesis_generation():
    """Demonstrate hypothesis generation from structure."""
    
    # Example: La2CuO4-type structure (simplied)
    structure = {
        'lattice': np.array([
            [3.8, 0, 0],
            [0, 3.8, 0],
            [0, 0, 13.2]
        ]),
        'species': ['Cu', 'Cu', 'O', 'O', 'O', 'O'],
        'coords': np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
    }
    
    # Generate candidates
    generator = GNNHypothesisGenerator(gnn_backend='heuristic')
    candidates = generator.generate_candidates(structure, max_candidates=4)
    
    print("Generated Candidate Hamiltonians:")
    print("=" * 60)
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n{i}. {candidate.name}")
        print(f"   Prior probability: {candidate.prior_probability:.2%}")
        print(f"   Description: {candidate.description}")
        print(f"   Parameters:")
        for param, value in candidate.exchange_terms.items():
            uncertainty = candidate.uncertainties.get(param, 0)
            print(f"      {param} = {value:.3f} ± {uncertainty:.3f} meV")
    
    return candidates


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo_hypothesis_generation()
