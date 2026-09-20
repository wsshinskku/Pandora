import numpy as np
import pytest
import torch

from pandora.calibration import RiskCalibrator
from pandora.config import ContractConfig, LearningConfig
from pandora.data import Episode, partition_episodes
from pandora.learning import ImpactModel, LocalLearner, ModelUpdate, federated_average


def episode(split="calibration", count=12):
    x = np.arange(count, dtype=float)[:, None]
    actions = np.tile([0.5, 1.0, 0.5], (count, 1))
    y = np.zeros((count, 5))
    y[-3:, 3] = 1
    return Episode(0, split, x, actions, actions.copy(), actions.copy(), y)


def test_calibration_quantile_and_leave_one_out():
    class Predictor:
        def predict(self, features):
            return np.zeros((len(features), 5)), np.full(len(features), 0.2)

    data = episode()
    cal = RiskCalibrator(ContractConfig()).fit(Predictor(), data)
    assert cal.threshold > 0  # A self-neighbor mistake would make this exactly zero.
    assert cal.margin == pytest.approx(0.8)
    assert cal.supported(data.features).all()
    far = data.features.copy()
    far[:, 0] += 1000
    assert not cal.supported(far).any()
    np.testing.assert_allclose(cal.upper_risk([0.5]), [1.0])
    with pytest.raises(ValueError):
        cal.fit(Predictor(), episode("test"))


def test_fedavg_sample_weighting_and_private_adapter():
    a, b = ImpactModel(4), ImpactModel(4)
    with torch.no_grad():
        a.adapter_bias.fill_(7)
        b.adapter_bias.fill_(9)
    first, second = a.shared_state(), b.shared_state()
    for key in first:
        first[key].fill_(2)
        second[key].fill_(6)
    average = federated_average([ModelUpdate(first, 1), ModelUpdate(second, 3)])
    assert all(torch.allclose(value, torch.full_like(value, 5)) for value in average.values())
    a.load_shared(average)
    assert torch.all(a.adapter_bias == 7)
    assert not any("adapter" in name for name in average)
    with pytest.raises(ValueError):
        federated_average([ModelUpdate(first, 0)])


def test_training_refuses_calibration_and_test_data():
    learner = LocalLearner(ImpactModel(4), LearningConfig(), 0.1, 3)
    for split in ("calibration", "test"):
        with pytest.raises(ValueError):
            learner.fit([episode(split)])
    with pytest.raises(ValueError):
        partition_episodes([episode("train"), episode("test")])


def test_training_changes_model_using_executed_action_and_no_personalization():
    torch.set_num_threads(1)
    model = ImpactModel(4, personalize=False)
    learner = LocalLearner(model, LearningConfig(epochs=1), 0.1, 3)
    data = episode("train")
    data.proposals[:] = 99
    before = model.kpi_head.weight.detach().clone()
    update, loss = learner.fit([data])
    assert update.samples == 12 and np.isfinite(loss)
    assert not torch.equal(before, model.kpi_head.weight)
    assert torch.count_nonzero(model.adapter_bias) == 0
    np.testing.assert_allclose(data.features[:, 1:], data.executed)


def test_episode_save_roundtrip(tmp_path):
    data = episode()
    data.save(tmp_path / "calibration.npz")
    restored = Episode.load(tmp_path / "calibration.npz")
    np.testing.assert_array_equal(data.features, restored.features)
    assert restored.split == "calibration"
