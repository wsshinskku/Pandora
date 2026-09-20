"""Train the same federated model from externally collected site-local NPZ episodes."""

from pathlib import Path

import torch

from .actions import ActionSpace
from .data import Episode, partition_episodes
from .experiment import train_models, write_json


def train_from_directory(config, data_directory, output, seed):
    config.validate()
    root, output = Path(data_directory), Path(output)
    datasets = []
    space = ActionSpace(config.simulation.users, config.simulation.cells)
    for site in range(config.simulation.sites):
        directory = root / f"site-{site}"
        paths = sorted(directory.glob("train-*.npz")) + sorted(directory.glob("calibration-*.npz"))
        episodes = [Episode.load(path) for path in paths]
        parts = partition_episodes(episodes)
        if (
            len(parts["train"]) != config.simulation.train_episodes
            or len(parts["calibration"]) != 1
        ):
            raise ValueError(
                f"site-{site}: expected configured training episodes and one calibration episode"
            )
        for episode in episodes:
            if episode.executed.shape[1] != space.dim:
                raise ValueError(f"site-{site}: action dimension does not match configuration")
            if not all(space.valid(action) for action in episode.executed):
                raise ValueError(f"site-{site}: invalid applied action in dataset")
        datasets.append(episodes)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Training output directory must be empty")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    models, calibrators, logs = train_models(config, datasets, seed)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "training.json", logs)
    for site, (model, calibrator) in enumerate(zip(models, calibrators)):
        local = output / f"site-{site}"
        local.mkdir()
        torch.save(
            {
                "input_dim": model.input_dim,
                "state_dict": model.state_dict(),
                "config": config.to_dict(),
                "site": site,
            },
            local / "pandora.pt",
        )
        calibration = partition_episodes(datasets[site])["calibration"][0]
        calibration.save(local / "calibration.npz")
        write_json(
            local / "calibration.json",
            {"margin": calibrator.margin, "support_threshold": calibrator.threshold},
        )
    return output
