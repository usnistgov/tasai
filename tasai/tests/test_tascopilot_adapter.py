from __future__ import annotations

from tasai.adapters.tascopilot import propose_next_point


def test_tascopilot_adapter_prefers_information_rate_candidate():
    result = propose_next_point(
        {
            "step": 0,
            "candidates": [
                {
                    "point_id": "slow",
                    "nodes": {"Q.H": -0.1, "Q.K": 0, "Q.L": 0, "et.deltaE": 5},
                    "metadata": {"posterior_variance": 10.0, "estimated_move_time": 1000.0},
                },
                {
                    "point_id": "informative",
                    "nodes": {"Q.H": 0.1, "Q.K": 0, "Q.L": 0, "et.deltaE": 5},
                    "metadata": {"posterior_variance": 8.0, "estimated_move_time": 1.0},
                    "count_time": 9.0,
                },
            ],
            "observations": [],
            "model_spec": {"eta": 0.7, "default_count_time": 12.0},
            "software_limits": {"Q.H": {"min": -1.0, "max": 1.0}},
            "policy_context": {"backend": "external_tasai_adapter"},
        }
    )

    assert result["point_id"] == "informative"
    assert result["reason"] == "tasai_uncertainty_information_rate"
    assert result["count_time"] == 9.0
