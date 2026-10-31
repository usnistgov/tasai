"""DREAM adapter tests: real BUMPS protocol, mocked sampling, no MCMC jobs."""
import os
from types import SimpleNamespace

# These mocked API tests do not exercise JIT performance. Disable Numba before
# optional Bumps DREAM imports so tests never write caches into site-packages.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pytest

from tasai.inference.mcmc import BumpsProblem, MCMCRunner


class LinearModel:
    """Small physics protocol fixture without optional spin-wave imports."""
    def __init__(self):
        self.free_parameters = [
            SimpleNamespace(name="scale", value=2.0, bounds=(0.1, 5.0)),
            SimpleNamespace(name="offset", value=1.0, bounds=(-3.0, 3.0)),
        ]
        self.n_free = 2

    def compute_intensity_array(self, h, k, l, E):
        return self.free_parameters[0].value * h + self.free_parameters[1].value

    def log_prior(self):
        if any(not p.bounds[0] <= p.value <= p.bounds[1] for p in self.free_parameters):
            return -np.inf
        offset = self.free_parameters[1].value
        return -0.5 * (offset / 0.5)**2 - np.log(0.5 * np.sqrt(2 * np.pi))

    def log_likelihood(self, h, k, l, E, I, sigma):
        residual = (I - self.compute_intensity_array(h, k, l, E)) / sigma
        return -0.5 * np.sum(residual**2) - np.sum(np.log(sigma * np.sqrt(2 * np.pi)))

    def log_posterior(self, *data):
        return self.log_prior() + self.log_likelihood(*data)

    def sample_prior(self, size):
        return np.column_stack((np.linspace(1.0, 3.0, size), np.linspace(-1.0, 1.0, size)))


def observed_data():
    h = np.arange(6, dtype=float)
    zero = np.zeros_like(h)
    return h, zero, zero, zero, 2 * h + 1, np.ones_like(h)


def test_real_fitproblem_protocol_sync_and_prior():
    FitProblem = pytest.importorskip("bumps.fitproblem").FitProblem
    model = LinearModel()
    fitness = BumpsProblem(model, *observed_data())
    problem = FitProblem(fitness)
    assert callable(fitness.parameters)
    assert problem.model_points() == 6
    assert problem.dof == 4
    problem.setp([3.0, 0.5])
    expected = model.log_posterior(*observed_data())
    assert problem.nllf() == pytest.approx(-expected)
    np.testing.assert_allclose(fitness.residuals(), 0.5 - np.arange(6))
    # residuals also synchronizes when parameters are edited directly.
    fitness.parameters()[0].value = 1.0
    np.testing.assert_allclose(fitness.residuals(), 0.5 + np.arange(6))


def test_dream_population_options_and_draw_contract(monkeypatch):
    core = pytest.importorskip("bumps.dream.core")
    captured = {}
    points = np.array([[2.0, 1.0], [2.5, 1.25]])
    logp = np.array([-4.0, -7.0])

    class FakeState:
        def mark_outliers(self):
            captured["marked"] = True

        def draw(self):
            return SimpleNamespace(points=points, logp=logp)

    class FakeDream(core.Dream):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

        def sample(self):
            # Exercise the actual DreamModel + FitProblem adapter without sampling.
            predicted = captured["model"].map(points)
            assert predicted.shape == (2,)
            assert np.isfinite(predicted).all()
            return FakeState()

    monkeypatch.setattr(core, "Dream", FakeDream)
    runner = MCMCRunner(LinearModel(), burn=2, steps=3, pop=4, backend="bumps")
    runner.set_data(*observed_data())
    initial = np.column_stack((np.linspace(1, 3, 10), np.linspace(-1, 1, 10)))
    chain = runner.run(initial_state=initial)
    np.testing.assert_allclose(captured["population"], initial[-8:][None, :, :])
    assert captured["draws"] == 8 * 3
    assert captured["burn"] == 8 * 2
    assert captured["thinning"] == 1
    assert captured["marked"]
    np.testing.assert_array_equal(chain, points)
    np.testing.assert_array_equal(runner.log_prob, logp)
    assert runner.dream_state is not None


def test_parallel_mapper_cleanup_after_sampler_error(monkeypatch):
    core = pytest.importorskip("bumps.dream.core")
    mapper_module = pytest.importorskip("bumps.mapper")
    events = []

    def start(problem, modelargs=None, cpus=0):
        events.append(("start", cpus))
        return lambda points: list(map(problem.nllf, points))

    def stop(mapper=None):
        events.append(("stop", mapper is not None))

    class FailingDream(core.Dream):
        def sample(self):
            raise RuntimeError("mock sampler failure")

    monkeypatch.setattr(core, "Dream", FailingDream)
    monkeypatch.setattr(mapper_module, "can_pickle", lambda problem: True)
    monkeypatch.setattr(mapper_module.MPMapper, "start_mapper", staticmethod(start))
    monkeypatch.setattr(mapper_module.MPMapper, "stop_mapper", staticmethod(stop))
    runner = MCMCRunner(LinearModel(), burn=1, steps=1, pop=4, parallel=True)
    runner.set_data(*observed_data())
    with pytest.raises(RuntimeError, match="mock sampler failure"):
        runner._run_bumps()
    assert events == [("start", 0), ("stop", True)]


def test_legacy_simple_entrypoint_delegates(monkeypatch):
    runner = MCMCRunner(LinearModel())
    initial = np.ones((8, 2))
    expected = np.ones((3, 2))
    def run(initial_state):
        assert initial_state is initial
        return expected
    monkeypatch.setattr(runner, "_run_bumps", run)
    assert runner._run_bumps_simple(initial) is expected


def test_order_parameter_example_uses_supported_population_keyword():
    # Validate real call-site keywords against the real constructor without
    # importing optional spin-wave backends through tasai.physics.__init__.
    import ast
    from pathlib import Path

    path = Path(__file__).parents[1] / "tasai/examples/andie_order_parameter.py"
    tree = ast.parse(path.read_text())
    populations = []
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "MCMCRunner":
            kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            runner = MCMCRunner(LinearModel(), **kwargs)
            if kwargs:
                populations.append(runner.pop)
    assert sorted(populations) == [4, 8]
