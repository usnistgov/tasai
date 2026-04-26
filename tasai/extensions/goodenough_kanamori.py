"""
Reusable Goodenough-Kanamori exchange-path analysis.

This module provides a structure-to-exchange-path analyzer that can be reused
by both the library hypothesis-generation stack and paper/demo scripts.

Convention
----------
The analyzer reports antiferromagnetic exchange as positive and
ferromagnetic exchange as negative, corresponding to:

    H = +J S_i . S_j
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_OCT_FILLING = {
    1: (1, 0, 0, 0),
    2: (2, 0, 0, 0),
    3: (3, 0, 0, 0),
    4: (3, 0, 1, 0),
    5: (3, 0, 2, 0),
    6: (3, 1, 2, 0),
    7: (3, 2, 2, 0),
    8: (3, 3, 2, 0),
    9: (3, 3, 2, 1),
}

_TET_FILLING = {
    1: (0, 0, 1, 0),
    2: (0, 0, 2, 0),
    3: (1, 0, 2, 0),
    4: (2, 0, 2, 0),
    5: (3, 0, 2, 0),
    6: (3, 0, 2, 1),
    7: (3, 0, 2, 2),
    8: (3, 1, 2, 2),
    9: (3, 2, 2, 2),
}

_GK_SIGMA = {
    ("half-filled", "half-filled"): ("AFM", 1.0),
    ("half-filled", "empty"): ("FM", 0.5),
    ("empty", "half-filled"): ("FM", 0.5),
    ("half-filled", "filled"): ("FM", 0.3),
    ("filled", "half-filled"): ("FM", 0.3),
    ("filled", "filled"): ("AFM", 0.1),
    ("empty", "empty"): ("AFM", 0.0),
    ("half-filled", "partial"): ("AFM", 0.6),
    ("partial", "half-filled"): ("AFM", 0.6),
    ("partial", "partial"): ("AFM", 0.4),
    ("filled", "partial"): ("FM", 0.2),
    ("partial", "filled"): ("FM", 0.2),
    ("filled", "empty"): ("AFM", 0.0),
    ("empty", "filled"): ("AFM", 0.0),
    ("partial", "empty"): ("FM", 0.2),
    ("empty", "partial"): ("FM", 0.2),
}

_JT_ACTIVE = {4, 9}

ION_CONFIGS = {
    "Fe3+": {"d_electrons": 5, "spin": 2.5},
    "Fe2+": {"d_electrons": 6, "spin": 2.0},
    "Fe": {"d_electrons": 5, "spin": 2.5},
    "Co2+": {"d_electrons": 7, "spin": 1.5},
    "Co": {"d_electrons": 7, "spin": 1.5},
    "Ni2+": {"d_electrons": 8, "spin": 1.0},
    "Ni": {"d_electrons": 8, "spin": 1.0},
    "Mn2+": {"d_electrons": 5, "spin": 2.5},
    "Mn3+": {"d_electrons": 4, "spin": 2.0},
    "Mn4+": {"d_electrons": 3, "spin": 1.5},
    "Mn": {"d_electrons": 5, "spin": 2.5},
    "Cr3+": {"d_electrons": 3, "spin": 1.5},
    "Cr": {"d_electrons": 3, "spin": 1.5},
    "Cu2+": {"d_electrons": 9, "spin": 0.5},
    "Cu": {"d_electrons": 9, "spin": 0.5},
    "V3+": {"d_electrons": 2, "spin": 1.0},
    "V": {"d_electrons": 2, "spin": 1.0},
}


def _subshell_status(n_up: int, n_dn: int) -> str:
    if n_up == 0 and n_dn == 0:
        return "empty"
    if n_dn == 0 and n_up > 0:
        return "half-filled"
    if n_up == n_dn and n_up > 0:
        return "filled"
    return "partial"


def orbital_status(d_count: int, coord: str = "oct") -> Dict[str, str]:
    table = _OCT_FILLING if coord == "oct" else _TET_FILLING
    if d_count not in table:
        return {"eg": "unknown", "t2g": "unknown"}
    t2g_up, t2g_dn, eg_up, eg_dn = table[d_count]
    return {
        "eg": _subshell_status(eg_up, eg_dn),
        "t2g": _subshell_status(t2g_up, t2g_dn),
    }


def _all_images_within(
    frac_i: np.ndarray,
    frac_j: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
    max_shell: int = 1,
) -> List[Tuple[float, np.ndarray, tuple]]:
    results: List[Tuple[float, np.ndarray, tuple]] = []
    frac_delta_base = frac_j - frac_i
    for na in range(-max_shell, max_shell + 1):
        for nb in range(-max_shell, max_shell + 1):
            for nc in range(-max_shell, max_shell + 1):
                frac_delta = frac_delta_base + np.array([na, nb, nc], dtype=float)
                cart_delta = frac_delta @ lattice
                dist = float(np.linalg.norm(cart_delta))
                if 0.01 < dist < cutoff:
                    results.append((dist, cart_delta, (na, nb, nc)))
    return results


@dataclass
class ExchangePath:
    site1: int
    site2: int
    site1_element: str
    site2_element: str
    distance: float
    image: Tuple[int, int, int] = (0, 0, 0)
    bridging_atoms: List[str] = field(default_factory=list)
    bridging_positions: List[np.ndarray] = field(default_factory=list)
    bond_angle: float = 180.0
    gk_channel: str = ""
    predicted_sign: str = "AFM"
    predicted_strength: float = 1.0
    confidence: float = 0.5
    validated: bool = False
    measured_J: Optional[float] = None
    measured_uncertainty: Optional[float] = None
    calibration_source: Optional[str] = None

    @property
    def path_type(self) -> str:
        if not self.bridging_atoms:
            return "direct"
        if len(self.bridging_atoms) == 1:
            return "superexchange"
        return "super-superexchange"

    @property
    def signed_strength(self) -> float:
        return self.predicted_strength if self.predicted_sign == "AFM" else -self.predicted_strength


class GoodenoughKanamoriAnalyzer:
    """Analyze exchange pathways from a crystal structure."""

    CALIBRATION_RULES = (
        {
            "name": "cuprate_180_superexchange",
            "description": "Cu-O-Cu 180° superexchange (cuprate reference)",
            "path_type": "superexchange",
            "elements": {"Cu"},
            "ligands": ("O",),
            "angle_min": 150,
            "angle_max": 190,
            "base_strength": 130.0,
            "distance_ref": 3.78,
            "distance_scale": 1.5,
            "confidence": 0.8,
            "force_sign": "AFM",
        },
    )

    def __init__(
        self,
        structure: Dict,
        calibrations: Optional[List[Dict]] = None,
        default_oxidation: Optional[Dict[str, str]] = None,
    ):
        self.structure = structure
        self.lattice = np.array(structure["lattice"], dtype=float)
        self.species = list(structure["species"])
        self.coords = np.array(structure["coords"], dtype=float)
        self.default_oxidation = default_oxidation or {}
        self.calibration_rules = tuple(calibrations or self.CALIBRATION_RULES)

        self.magnetic_sites: List[int] = []
        self.ligand_sites: List[int] = []
        for idx, elem in enumerate(self.species):
            if self._is_magnetic(elem):
                self.magnetic_sites.append(idx)
            elif self._is_ligand(elem):
                self.ligand_sites.append(idx)

    @staticmethod
    def _is_magnetic(element: str) -> bool:
        return element in {
            "Fe", "Co", "Ni", "Mn", "Cr", "Cu", "V",
            "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        }

    @staticmethod
    def _is_ligand(element: str) -> bool:
        return element in {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}

    def _ion_key(self, element: str) -> str:
        return self.default_oxidation.get(element, element)

    def _d_count(self, element: str) -> int:
        cfg = ION_CONFIGS.get(self._ion_key(element))
        return -1 if cfg is None else cfg["d_electrons"]

    def _coord_number(self, site_idx: int, bond_cutoff: float = 2.8) -> int:
        frac_i = self.coords[site_idx]
        count = 0
        for lig_idx in self.ligand_sites:
            images = _all_images_within(frac_i, self.coords[lig_idx], self.lattice, bond_cutoff)
            count += len(images)
        return count

    def _guess_coordination(self, site_idx: int) -> str:
        coord_number = self._coord_number(site_idx)
        if coord_number >= 5:
            return "oct"
        if coord_number <= 3:
            return "tet"

        frac_i = self.coords[site_idx]
        cart_i = frac_i @ self.lattice
        ligand_z: List[float] = []
        for lig_idx in self.ligand_sites:
            images = _all_images_within(frac_i, self.coords[lig_idx], self.lattice, 2.8)
            for _, delta, _ in images:
                ligand_z.append((cart_i + delta)[2])
        if ligand_z and max(ligand_z) - min(ligand_z) < 1.0:
            return "oct"
        return "tet"

    def find_exchange_paths(
        self,
        max_distance: float = 8.0,
        max_bridging: int = 2,
        bond_cutoff: float = 2.8,
    ) -> List[ExchangePath]:
        paths: List[ExchangePath] = []
        seen = set()

        for left_idx, site_i in enumerate(self.magnetic_sites):
            for right_idx, site_j in enumerate(self.magnetic_sites):
                if right_idx <= left_idx and site_i != site_j:
                    continue

                frac_i = self.coords[site_i]
                frac_j = self.coords[site_j]
                images = _all_images_within(frac_i, frac_j, self.lattice, max_distance)
                for distance, cart_delta_ij, image in images:
                    key = (min(site_i, site_j), max(site_i, site_j), image)
                    reverse_key = (min(site_i, site_j), max(site_i, site_j), tuple(-x for x in image))
                    if key in seen or reverse_key in seen:
                        continue
                    seen.add(key)

                    cart_i = frac_i @ self.lattice
                    cart_j_img = cart_i + cart_delta_ij
                    bridging_atoms: List[str] = []
                    bridging_positions: List[np.ndarray] = []

                    for lig_idx in self.ligand_sites:
                        lig_images = _all_images_within(frac_i, self.coords[lig_idx], self.lattice, bond_cutoff)
                        for _, delta_ik, _ in lig_images:
                            cart_k_img = cart_i + delta_ik
                            d_kj = float(np.linalg.norm(cart_k_img - cart_j_img))
                            if d_kj < bond_cutoff:
                                bridging_atoms.append(self.species[lig_idx])
                                bridging_positions.append(cart_k_img)

                    if len(bridging_atoms) > max_bridging:
                        midpoint = (cart_i + cart_j_img) / 2
                        order = np.argsort([np.linalg.norm(pos - midpoint) for pos in bridging_positions])[:max_bridging]
                        bridging_atoms = [bridging_atoms[idx] for idx in order]
                        bridging_positions = [bridging_positions[idx] for idx in order]

                    if bridging_positions:
                        v1 = cart_i - bridging_positions[0]
                        v2 = cart_j_img - bridging_positions[0]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1.0e-10)
                        angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                    else:
                        angle = 180.0

                    path = ExchangePath(
                        site1=site_i,
                        site2=site_j,
                        site1_element=self.species[site_i],
                        site2_element=self.species[site_j],
                        distance=distance,
                        image=image,
                        bridging_atoms=bridging_atoms,
                        bridging_positions=bridging_positions,
                        bond_angle=angle,
                    )
                    self._apply_gk_rules(path)
                    paths.append(path)

        return paths

    def _apply_gk_rules(self, path: ExchangePath) -> None:
        d1 = self._d_count(path.site1_element)
        d2 = self._d_count(path.site2_element)
        coord1 = self._guess_coordination(path.site1)
        coord2 = self._guess_coordination(path.site2)
        orb1 = orbital_status(d1, coord1) if d1 > 0 else {"eg": "unknown", "t2g": "unknown"}
        orb2 = orbital_status(d2, coord2) if d2 > 0 else {"eg": "unknown", "t2g": "unknown"}
        jt_active = (d1 in _JT_ACTIVE) or (d2 in _JT_ACTIVE)

        if path.path_type == "direct":
            path.predicted_sign = "AFM"
            path.predicted_strength = 0.5 * np.exp(-path.distance / 3.0)
            path.confidence = 0.3
            path.gk_channel = f"direct d{d1}-d{d2}"
        elif path.path_type == "superexchange":
            sigma_key = (orb1["eg"], orb2["eg"])
            sigma_sign, sigma_scale = _GK_SIGMA.get(sigma_key, ("AFM", 0.3))
            sigma_weight = np.cos(np.radians(180.0 - path.bond_angle)) ** 2
            pi_weight = np.sin(np.radians(path.bond_angle)) ** 2
            j_sigma = sigma_scale * sigma_weight
            j_pi = 0.15 * pi_weight

            if sigma_sign == "AFM":
                j_net = j_sigma - j_pi
                path.predicted_sign = "AFM" if j_net >= 0 else "FM"
                path.predicted_strength = abs(j_net) * 10.0
            else:
                path.predicted_sign = "FM"
                path.predicted_strength = (j_sigma + j_pi) * 10.0

            path.predicted_strength *= np.exp(-(path.distance - 3.5) / 2.0)
            if "unknown" in sigma_key:
                path.confidence = 0.3
            elif jt_active:
                path.confidence = 0.5
            else:
                path.confidence = 0.75
            path.gk_channel = (
                f"σ: eg {orb1['eg']}/{orb2['eg']} -> {sigma_sign} "
                f"(scale {sigma_scale:.2f}), angle {path.bond_angle:.0f}°"
            )
        else:
            path.predicted_sign = "AFM"
            path.predicted_strength = 0.5 * np.exp(-path.distance / 4.0)
            path.confidence = 0.25
            path.gk_channel = f"super-SE d{d1}-d{d2}, {len(path.bridging_atoms)} bridges"

        self._apply_calibration_overrides(path)

    def _apply_calibration_overrides(self, path: ExchangePath) -> None:
        ligand_counter = Counter(path.bridging_atoms)
        for rule in self.calibration_rules:
            if rule.get("path_type") and rule["path_type"] != path.path_type:
                continue
            rule_elements = rule.get("elements")
            if rule_elements and (
                path.site1_element not in rule_elements or path.site2_element not in rule_elements
            ):
                continue
            ligands = rule.get("ligands")
            if ligands and Counter(ligands) != ligand_counter:
                continue
            if rule.get("angle_min") and path.bond_angle < rule["angle_min"]:
                continue
            if rule.get("angle_max") and path.bond_angle > rule["angle_max"]:
                continue

            base = rule.get("base_strength", path.predicted_strength)
            distance_ref = rule.get("distance_ref", path.distance)
            distance_scale = rule.get("distance_scale", 3.0)
            path.predicted_strength = max(base * np.exp(-(path.distance - distance_ref) / distance_scale), 0.0)
            path.confidence = rule.get("confidence", path.confidence)
            force_sign = rule.get("force_sign")
            if force_sign:
                path.predicted_sign = force_sign
            path.calibration_source = rule.get("name")
            break

    def rank_paths(self, paths: List[ExchangePath]) -> List[ExchangePath]:
        for path in paths:
            path._score = path.predicted_strength * path.confidence
        return sorted(paths, key=lambda item: -item._score)

    def cluster_paths(
        self,
        paths: List[ExchangePath],
        distance_tol: float = 0.05,
        angle_tol: float = 1.0,
    ) -> List[List[ExchangePath]]:
        clusters: List[List[ExchangePath]] = []

        def is_equivalent(left: ExchangePath, right: ExchangePath) -> bool:
            if left.path_type != right.path_type:
                return False
            if abs(left.distance - right.distance) > distance_tol:
                return False
            if abs(left.bond_angle - right.bond_angle) > angle_tol:
                return False
            return Counter(left.bridging_atoms) == Counter(right.bridging_atoms)

        for path in self.rank_paths(paths):
            for cluster in clusters:
                if is_equivalent(path, cluster[0]):
                    cluster.append(path)
                    break
            else:
                clusters.append([path])
        return clusters

    def generate_hamiltonians(self, paths: List[ExchangePath], max_terms: int = 3) -> List[Dict]:
        del max_terms
        clusters = self.cluster_paths(paths)
        reps = [cluster[0] for cluster in clusters]
        candidates: List[Dict] = []

        if reps:
            candidates.append(
                {
                    "name": "Single-J",
                    "terms": {"J1": reps[0].signed_strength},
                    "paths_used": [0],
                    "multiplicities": [len(clusters[0])],
                    "description": f"{reps[0].path_type} via {reps[0].bridging_atoms}, {reps[0].gk_channel}",
                    "prior": 0.3,
                }
            )
        if len(reps) >= 2:
            candidates.append(
                {
                    "name": "Two-J",
                    "terms": {"J1": reps[0].signed_strength, "J2": reps[1].signed_strength},
                    "paths_used": [0, 1],
                    "multiplicities": [len(clusters[0]), len(clusters[1])],
                    "description": (
                        f"J1: {reps[0].path_type} ({reps[0].gk_channel}), "
                        f"J2: {reps[1].path_type} ({reps[1].gk_channel})"
                    ),
                    "prior": 0.5,
                }
            )
            candidates.append(
                {
                    "name": "Two-J + D",
                    "terms": {"J1": reps[0].signed_strength, "J2": reps[1].signed_strength, "D": 0.1},
                    "paths_used": [0, 1],
                    "multiplicities": [len(clusters[0]), len(clusters[1])],
                    "description": "Two-J model with single-ion anisotropy placeholder",
                    "prior": 0.2,
                }
            )
        return candidates

