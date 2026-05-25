"""TAS co-pilot adaptive policy adapter for tasAI.

This module keeps policy scoring in the tasAI repository while exposing the
JSON-compatible boundary expected by tascopilot:

    propose_next_point(payload: dict) -> dict

The payload and return value are validated with tas-copilot-contracts when that
package is installed. Heavy MCMC/bumps policy updates should run in this tasAI
environment or a TACC/container runtime, not inside tascopilot core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from tas_copilot_contracts import AdaptivePolicyProposal, AdaptivePolicyRequest
except ImportError as exc:  # pragma: no cover - depends on integration environment
    AdaptivePolicyProposal = None  # type: ignore[assignment]
    AdaptivePolicyRequest = None  # type: ignore[assignment]
    _CONTRACT_IMPORT_ERROR = exc
else:
    _CONTRACT_IMPORT_ERROR = None


def propose_next_point(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the next adaptive point using tasAI-style information-rate scoring.

    Candidate hints are supplied in `candidate.metadata`:
    - posterior_variance
    - expected_information_gain
    - estimated_move_time
    - forecast_step

    The scoring is deliberately light enough for a local simulator loop. Full
    HH/forecast/MCMC policies should reuse this same contract boundary and run
    in a tasAI-owned runtime.
    """

    if AdaptivePolicyRequest is None or AdaptivePolicyProposal is None:
        raise RuntimeError(
            "tas-copilot-contracts is required for tasAI/tascopilot adapter validation"
        ) from _CONTRACT_IMPORT_ERROR

    request = AdaptivePolicyRequest.model_validate(payload)
    if not request.candidates:
        proposal = AdaptivePolicyProposal(
            point_id="no-candidate",
            nodes={},
            score=0.0,
            reason="missing_candidates",
            metadata={"adapter": "tasai.adapters.tascopilot"},
        )
        return proposal.model_dump(mode="json")

    eta = float(request.model_spec.get("eta") or 0.7)
    motion_penalty_weight = float(request.model_spec.get("motion_penalty_weight") or 1.0)
    default_count_time = float(request.model_spec.get("default_count_time") or 60.0)

    best = None
    best_score = -np.inf
    best_components: dict[str, float] = {}
    for index, candidate in enumerate(request.candidates):
        metadata = candidate.metadata or {}
        variance = float(metadata.get("posterior_variance") or metadata.get("variance") or 0.0)
        info_gain = float(metadata.get("expected_information_gain") or variance)
        move_time = float(metadata.get("estimated_move_time") or metadata.get("move_time") or 0.0)
        count_time = float(candidate.count_time or metadata.get("count_time") or default_count_time)
        forecast_step = float(metadata.get("forecast_step") or 0.0)
        forecast_bonus = 1.0 / (1.0 + max(forecast_step, 0.0))
        numerator = max(info_gain, 0.0) ** eta + max(variance, 0.0)
        denominator = max(count_time + motion_penalty_weight * move_time, 1.0)
        score = forecast_bonus * numerator / denominator - index * 1e-12
        if score > best_score:
            best = candidate
            best_score = float(score)
            best_components = {
                "posterior_variance": variance,
                "expected_information_gain": info_gain,
                "estimated_move_time": move_time,
                "count_time": count_time,
                "forecast_bonus": forecast_bonus,
            }

    assert best is not None
    proposal = AdaptivePolicyProposal(
        point_id=best.point_id,
        nodes=best.nodes,
        score=best_score,
        reason="tasai_uncertainty_information_rate",
        count_time=best_components["count_time"],
        metadata={
            "adapter": "tasai.adapters.tascopilot",
            "policy_family": "tasAI_information_rate_preview",
            "components": best_components,
            "behavior_boundary": "planning_or_simulation_only",
        },
    )
    return proposal.model_dump(mode="json")

