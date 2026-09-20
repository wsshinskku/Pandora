import json

import numpy as np
import pytest

from pandora.config import Config, SimulationConfig, load_config
from pandora.experiment import run_experiment
from pandora.integration import JsonlSession, load_controller
from pandora.metrics import bootstrap_mean, episode_metrics, paired_difference, prediction_metrics
from pandora.simulator import AnalyticalRAN
from pandora.traces import import_sinr_csv
from pandora.xapps import ReferenceXApps


def tiny_config():
    config = Config(
        simulation=SimulationConfig(sites=2, users=3, cells=2, slots=12, train_episodes=1),
        seeds=[1, 2],
        bootstrap_samples=100,
    )
    config.learning.epochs = 1
    config.learning.fl_period = 12
    return config


def test_exogenous_traces_are_action_independent():
    cfg = SimulationConfig(users=3, slots=10)
    a, b = AnalyticalRAN(cfg, 1, 0, 0), AnalyticalRAN(cfg, 1, 0, 0)
    np.testing.assert_array_equal(a.arrivals, b.arrivals)
    np.testing.assert_array_equal(a.capacities, b.capacities)
    a.step(a.space.neutral())
    action = b.space.neutral()
    action[:3] = 0
    b.step(action)
    np.testing.assert_array_equal(a.observe().demand, b.observe().demand)
    np.testing.assert_array_equal(a.observe().capacity, b.observe().capacity)


def test_metrics_tail_direction_and_single_class_auc():
    y = np.column_stack([np.arange(1, 11), np.arange(1, 11), np.zeros((10, 3))])
    metrics = episode_metrics(y, 0.1)
    assert metrics["throughput_top10_mbps"] == 10
    assert metrics["delay_top10_ms"] == 1
    assert metrics["delay_bottom10_ms"] == 10
    assert prediction_metrics(y, y, np.zeros(10), 0.1)["risk_auroc"] is None
    assert bootstrap_mean([1], np.random.default_rng(1))["ci95"] is None
    assert paired_difference({0: 1, 1: 2}, {0: 2, 1: 3}, np.random.default_rng(1))["mean"] == 1
    with pytest.raises(ValueError):
        paired_difference({0: 1}, {1: 2}, np.random.default_rng(1))


def test_trace_import_rejects_missing_and_duplicate_samples(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("slot,ue,cell,sinr_db\n0,0,0,0\n", encoding="utf-8")
    np.testing.assert_allclose(import_sinr_csv(path, 1, 1, 1, 20), [[[20]]])
    with pytest.raises(ValueError, match="incomplete"):
        import_sinr_csv(path, 2, 1, 1, 20)
    path.write_text("slot,ue,cell,sinr_db\n0,0,0,0\n0,0,0,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        import_sinr_csv(path, 1, 1, 1, 20)


def test_config_rejects_invalid_and_unknown_settings(tmp_path):
    config = tiny_config()
    config.contract.weights[0] = 0
    with pytest.raises(ValueError):
        config.validate()
    path = tmp_path / "invalid.yaml"
    path.write_text("simulation:\n  nonexistent: true", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)


def test_saved_json_config_roundtrip_handles_scientific_notation(tmp_path):
    path = tmp_path / "config.json"
    config = tiny_config()
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path).to_dict() == config.to_dict()


def test_adaptive_bounds_handle_small_solver_residuals():
    from pandora.actions import ActionSpace
    from pandora.baselines import BaselineController
    from pandora.contracts import fallback_contract

    space = ActionSpace(2, 1)
    fallback = fallback_contract(space, [], np.empty((0, space.dim)), np.empty(0))
    baseline = BaselineController("adaptive", space, tiny_config().contract, fallback)
    action = space.neutral()
    action[-1] = -1e-7
    baseline.feedback(action, [1, 1, 0, 0, 1])
    baseline.feedback(action, [1, 1, 0, 0, 1])
    assert space.valid(baseline.decide(space.neutral(), 30, None))


def test_complete_run_checkpoint_and_acknowledgement_protocol(tmp_path):
    config = tiny_config()
    output = tmp_path / "run"
    result = run_experiment(
        config, output, ["independent", "adaptive", "pandora"], progress=lambda _: None
    )
    assert set(result) == {"independent", "adaptive", "pandora"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["paper_results_reproduced"]
    assert result["pandora"]["fairness"]["ci95"] is not None
    environment = AnalyticalRAN(config.simulation, 1, 0, 2)
    local = output / "seed-1" / "site-0"
    controller = load_controller(
        local / "pandora.pt", local / "calibration-1.npz", environment.slices
    )
    audit = tmp_path / "audit.jsonl"
    session = JsonlSession(controller, audit)
    telemetry = environment.observe()
    proposal = ReferenceXApps(environment.space, config.simulation).propose(telemetry)
    request = {
        "version": 1,
        "slot": 0,
        "observation": telemetry.vector(config.simulation.buffer_mbit).tolist(),
        "proposal": proposal.tolist(),
    }
    response = session.handle(request)
    assert response == session.handle(request) and session.slot == 1
    # A deliberately different executed action must be retained in the audit record.
    applied = environment.space.neutral()
    y = environment.step(applied)
    with pytest.raises(ValueError, match="Feedback"):
        session.handle({"version": 1, "slot": 1, "type": "finish"})
    done = session.handle(
        {
            "version": 1,
            "slot": 1,
            "type": "finish",
            "feedback": {"slot": 0, "executed": applied.tolist(), "kpis": y.tolist()},
        }
    )
    assert done["finished"]
    record = json.loads(audit.read_text())
    np.testing.assert_array_equal(record["executed"], applied)
    np.testing.assert_array_equal(record["projected"], response["action"])
    with pytest.raises(ValueError, match="empty"):
        run_experiment(config, output)
