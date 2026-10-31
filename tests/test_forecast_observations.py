"""Offline regression checks for the existing mean-outcome forecast policy."""
from dataclasses import dataclass

import numpy as np
import pytest

from tasai.core.forecast import (
    ExperimentRunner, Forecaster, ForecastStopReason,
)


@dataclass
class Candidate:
    E: float = 150.0
    count_time: float = 60.0
    duration: float = 1200.0


class PairAdapter:
    def __init__(self, covariance=None):
        self.matrix = np.array([[4.0, 1.0], [1.0, 9.0]]) if covariance is None else covariance
        self.seen_shapes = []

    def predict(self, samples, candidate):
        self.seen_shapes.append(samples.shape)
        # Full nuisance columns must survive even when entropy targets Tc only.
        return np.column_stack([samples[:, 0] / 100 + samples[:, 1],
                                2 * samples[:, 0] / 100 + samples[:, 2]])

    def covariance(self, candidate, count_time, predictions, weights):
        return self.matrix

    def count_time(self, candidate, predictions, weights):
        return candidate.duration


def particles():
    rng = np.random.default_rng(17)
    return np.column_stack([rng.uniform(100, 200, 240),
                            rng.normal(0, .3, 240), rng.normal(0, .3, 240)])


def make_forecaster(adapter, candidate=None, **kwargs):
    candidate = candidate or Candidate()
    return Forecaster(None, lambda: [candidate], poi_indices=[0],
                      observation_adapter=adapter, rng=np.random.default_rng(52), **kwargs)


def test_joint_gaussian_uses_full_covariance_and_full_nuisance_particles():
    adapter = PairAdapter()
    forecaster = make_forecaster(adapter)
    samples = np.array([[100., 1., 3.], [150., 2., 6.], [200., 3., 9.]])
    weights = np.array([.25, .5, .25])
    prediction, mean, covariance, duration, log_like = forecaster._vector_observation(
        samples, Candidate(), weights)
    np.testing.assert_allclose(mean, [3.5, 9.])
    np.testing.assert_allclose(log_like, [-72.25 / 70, 0., -72.25 / 70])
    np.testing.assert_allclose(covariance, [[4., 1.], [1., 9.]])
    assert duration == 1200.0  # Complete scans are not clipped to scalar600s maximum.
    assert adapter.seen_shapes == [(3, 3)]
    diagonal = make_forecaster(PairAdapter(np.diag([4., 9.])))
    diagonal_log_like = diagonal._vector_observation(samples, Candidate(), weights)[-1]
    assert not np.allclose(log_like, diagonal_log_like)
    changed = samples.copy(); changed[0, 2] += 10
    assert not np.allclose(forecaster._vector_observation(changed, Candidate(), weights)[-1], log_like)


def test_seeded_pair_forecast_preserves_actual_posterior_and_targets_only_tc(monkeypatch):
    from tasai.core import entropy
    original_entropy = entropy.differential_entropy
    entropy_shapes = []

    def record_entropy(samples, **kwargs):
        entropy_shapes.append(samples.shape)
        return original_entropy(samples, **kwargs)

    monkeypatch.setattr(entropy, 'differential_entropy', record_entropy)
    samples = particles(); actual = samples.copy()
    global_state = np.random.get_state()
    adapter = PairAdapter(np.array([[.01, .003], [.003, .02]]))
    result = make_forecaster(adapter).forecast(samples, n_forecast=2, min_info_gain=-1)
    repeat = make_forecaster(PairAdapter(adapter.matrix)).forecast(samples, n_forecast=2, min_info_gain=-1)
    np.testing.assert_array_equal(samples, actual)
    np.testing.assert_array_equal(global_state[1], np.random.get_state()[1])
    assert global_state[2:] == np.random.get_state()[2:]
    assert result.posterior_kind == 'hypothetical_mean_outcome'
    assert result.hypothetical_posterior_samples is result.posterior_samples
    assert not np.shares_memory(result.posterior_samples, samples)
    assert result.posterior_samples.shape[1] == 3
    assert all(shape[1] == 3 for shape in adapter.seen_shapes)
    assert all(shape[1] == 1 for shape in entropy_shapes)
    np.testing.assert_array_equal(result.posterior_samples, repeat.posterior_samples)
    assert [p.score for p in result.points] == [p.score for p in repeat.points]
    assert result.points
    np.testing.assert_allclose(result.points[0].expected_covariance, adapter.matrix)


def test_full_pair_duration_and_motion_are_charged_once():
    samples = particles()
    adapter = PairAdapter(np.diag([.01, .02]))
    fast = make_forecaster(adapter, Candidate(duration=1200.),
                           movement_time_func=lambda a, b: 300.)
    slow = make_forecaster(adapter, Candidate(duration=2400.),
                           movement_time_func=lambda a, b: 300.)
    r_fast = fast.forecast(samples, n_forecast=1, current_position=Candidate())
    r_slow = slow.forecast(samples, n_forecast=1, current_position=Candidate())
    assert len(r_fast.points) == len(r_slow.points) == 1
    assert r_fast.total_info_gain == r_slow.total_info_gain
    f, s = r_fast.points[0], r_slow.points[0]
    assert f.count_time == 1200. and f.movement_time == 300.
    assert f.score / s.score == pytest.approx(2700. / 1500.)



def test_vector_startup_cost_uses_callback_even_without_previous_position():
    calls = []

    def movement(previous, candidate):
        calls.append(previous)
        return 450.

    result = make_forecaster(PairAdapter(np.diag([.01, .02])),
                              movement_time_func=movement).forecast(particles(), n_forecast=1)
    assert calls == [None]
    assert result.points[0].movement_time == 450.


def test_low_ess_stops_without_accepting_update_or_resetting_actual_particles():
    class NarrowAdapter(PairAdapter):
        def predict(self, samples, candidate):
            return samples[:, :1]

    samples = np.column_stack([np.linspace(100, 200, 51), np.zeros((51, 2))])
    before = samples.copy()
    result = make_forecaster(NarrowAdapter(np.array([[1e-12]]))).forecast(samples)
    assert result.stop_reason is ForecastStopReason.LOW_EFFECTIVE_SAMPLE_SIZE
    assert result.requires_posterior_refresh
    assert result.stop_details['effective_sample_size'] == pytest.approx(1.)
    assert result.points == []
    assert result.total_info_gain == 0.
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(result.hypothetical_posterior_samples, before)


@pytest.mark.parametrize('matrix', [np.eye(3), [[1., 1.], [1., 1.]],
                                    [[1., .5], [0., 1.]], [[np.nan, 0.], [0., 1.]]])
def test_invalid_observation_covariance_fails_explicitly(matrix):
    with pytest.raises(ValueError, match='covariance'):
        make_forecaster(PairAdapter(np.asarray(matrix))).forecast(particles(), n_forecast=1)


def test_invalid_vector_shape_and_duration_fail_explicitly():
    class BadShape(PairAdapter):
        def predict(self, samples, candidate):
            return np.zeros(len(samples))

    with pytest.raises(ValueError, match='predict'):
        make_forecaster(BadShape()).forecast(particles())
    with pytest.raises(ValueError, match='count_time'):
        make_forecaster(PairAdapter(), Candidate(duration=0.)).forecast(particles())


def test_scalar_policy_historical_golden_result_is_unchanged():
    class ScalarModel:
        def samples_to_params(self, p):
            return {'x': p[0], 'nuisance': p[1]}

        def set_parameters(self, params):
            self.params = params

        def compute_intensity(self, h, k, l, energy):
            return 20 + self.params['x'] + .2 * self.params['nuisance']

    samples = np.column_stack([np.linspace(-3, 3, 80), np.sin(np.linspace(0, 4, 80))])
    saved_state = np.random.get_state()
    try:
        np.random.seed(241)
        result = Forecaster(ScalarModel(), lambda: [Candidate()], poi_indices=[0]).forecast(
            samples, n_forecast=1)
    finally:
        np.random.set_state(saved_state)
    assert len(result.points) == 1
    point = result.points[0]
    # Generated using this repository's unmodified5070550f Forecaster, not the new vector path.
    assert point.count_time == 10.
    assert point.score == pytest.approx(0.009104300943517869)
    assert result.total_info_gain == pytest.approx(0.23620829103839047)


def test_experiment_runner_imports_inference_mcmc_without_running_sampler(monkeypatch):
    from tasai.inference import mcmc
    captured = {}

    class FakeMCMC:
        def __init__(self, model, **kwargs):
            captured['kwargs'] = kwargs

        def run(self, *arrays):
            captured['arrays'] = arrays
            return np.array([[140., 1., 2.], [160., 2., 3.]])

    monkeypatch.setattr(mcmc, 'MCMCRunner', FakeMCMC)
    runner = ExperimentRunner(object(), object(), lambda: [], poi_indices=[0])
    runner.add_measurement(Candidate(), 10., .5)
    posterior = runner.run_inference()
    assert posterior.shape == (2, 3)
    assert captured['arrays'][3].tolist() == [150.]
    assert captured['arrays'][4].tolist() == [10.]
