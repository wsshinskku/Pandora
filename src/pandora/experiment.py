"""End-to-end local experiment runner with paired exogenous traces and isolated test episodes."""

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .baselines import BaselineController
from .calibration import RiskCalibrator
from .contracts import fallback_contract
from .controller import PandoraController
from .data import Episode, partition_episodes
from .learning import ImpactModel, LocalLearner, federated_average
from .metrics import bootstrap_mean, episode_metrics, paired_difference, prediction_metrics
from .simulator import AnalyticalRAN
from .synthesis import ContractSynthesizer
from .xapps import ReferenceXApps

METHODS = (
    "independent",
    "static",
    "adaptive",
    "scheduler-reference",
    "qos-reference",
    "pandora",
    "pandora-no-personalization",
    "pandora-no-fl",
    "pandora-no-calibration",
    "pandora-box-only",
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def collect_episode(config, seed, site, episode_id, split):
    environment = AnalyticalRAN(config.simulation, seed, site, episode_id)
    apps = ReferenceXApps(environment.space, config.simulation)
    rng = np.random.default_rng(np.random.SeedSequence([seed, site, episode_id, 923]))
    observations, proposals, applied, kpis = [], [], [], []
    for _ in range(config.simulation.slots):
        telemetry = environment.observe()
        proposal = apps.propose(telemetry)
        # Exploration is used only for the pre-collected train/calibration episodes.
        # It is deliberately not optimized using the outcomes of held-out episodes.
        if rng.random() < 0.15:
            action = environment.space.sample(rng, 1)[0]
        else:
            action = environment.space.normalize(proposal + rng.normal(0, 0.015, proposal.shape))
        observations.append(telemetry.vector(config.simulation.buffer_mbit))
        proposals.append(proposal)
        applied.append(action)
        kpis.append(environment.step(action))
    return Episode(
        episode_id,
        split,
        np.array(observations),
        np.array(proposals),
        np.array(applied),
        np.array(applied),
        np.array(kpis),
    )


def train_models(config, datasets, seed):
    """Synchronous in-process FL; the aggregator receives ModelUpdate objects only."""
    torch.manual_seed(seed)
    input_dim = datasets[0][0].features.shape[1]
    template = ImpactModel(input_dim, config.learning.personalize)
    models = [copy.deepcopy(template) for _ in datasets]
    learners = [
        LocalLearner(m, config.learning, config.contract.epsilon, seed + i)
        for i, m in enumerate(models)
    ]
    train, calibration = [], []
    for site_data in datasets:
        parts = partition_episodes(site_data)
        if parts["test"] or len(parts["calibration"]) != 1 or not parts["train"]:
            raise ValueError(
                "Training expects train episodes and exactly one local calibration episode"
            )
        arrays = {
            name: np.concatenate([getattr(e, name) for e in parts["train"]])
            for name in ("observations", "proposals", "projected", "executed", "kpis")
        }
        train.append(arrays)
        calibration.append(parts["calibration"][0])
    lengths = [len(a["kpis"]) for a in train]
    if len(set(lengths)) != 1:
        raise ValueError("This synchronous runner requires equal training horizons at all sites")
    logs = []
    calibrators = []
    for round_id, start in enumerate(range(0, lengths[0], config.learning.fl_period)):
        updates, losses = [], []
        for learner, arrays in zip(learners, train):
            end = start + config.learning.fl_period
            chunk = Episode(round_id, "train", **{k: v[start:end] for k, v in arrays.items()})
            update, loss = learner.fit([chunk])
            updates.append(update)
            losses.append(loss)
        if config.learning.federate:
            shared = federated_average(updates)
            for model in models:
                model.load_shared(shared)
        calibrators = [
            RiskCalibrator(config.contract, config.learning.calibrate).fit(m, c)
            for m, c in zip(models, calibration)
        ]
        logs.append(
            {
                "round": round_id + 1,
                "samples": [u.samples for u in updates],
                "loss": losses,
                "margins": [c.margin for c in calibrators],
                "upload_bytes": [
                    sum(t.numel() * t.element_size() for t in u.shared.values())
                    if config.learning.federate
                    else 0
                    for u in updates
                ],
            }
        )
    return models, calibrators, logs


def method_config(config, method):
    c = copy.deepcopy(config)
    changes = {
        "pandora-no-personalization": "personalize",
        "pandora-no-fl": "federate",
        "pandora-no-calibration": "calibrate",
        "pandora-box-only": "coupling",
    }
    if method in changes:
        setattr(c.learning, changes[method], False)
    return c


def build_controller(config, model, calibrator, calibration, environment, seed):
    space = environment.space
    H, limits = space.coupling_template(environment.slices)
    if not config.learning.coupling:
        H, limits = np.empty((0, space.dim)), np.empty(0)
    fallback = fallback_contract(space, calibration.executed, H, limits)
    synthesizer = ContractSynthesizer(
        space, config.contract, model, calibrator, H, limits, fallback, seed
    )
    return PandoraController(synthesizer)


def evaluate_site(config, method, seed, site, calibration, model=None, calibrator=None):
    episode_id = config.simulation.train_episodes + 1
    env = AnalyticalRAN(config.simulation, seed, site, episode_id)
    apps = ReferenceXApps(env.space, config.simulation)
    if method.startswith("pandora"):
        controller = build_controller(
            config, model, calibrator, calibration, env, seed * 1000 + site + 17
        )
    else:
        H, limits = env.space.coupling_template(env.slices)
        fallback = fallback_contract(env.space, calibration.executed, H, limits)
        controller = BaselineController(method, env.space, config.contract, fallback)
    obs, proposals, actions, outcomes, rows, syntheses = [], [], [], [], [], []
    for slot in range(config.simulation.slots):
        telemetry = env.observe()
        observation = telemetry.vector(config.simulation.buffer_mbit)
        proposal = apps.propose(telemetry)
        if method.startswith("pandora"):
            decision = controller.decide(observation, proposal, slot)
            action = decision.action
            _, risk = model.predict(np.r_[observation, proposal][None])
            info = {
                "intervention": decision.intervention,
                "fallback": decision.fallback,
                "projection_distance": decision.distance,
                "normalized_distance": decision.normalized_distance,
                "supported": decision.supported,
                "proposal_risk": float(risk[0]),
                "reason": decision.reason,
            }
            if decision.synthesis:
                syntheses.append(decision.synthesis)
        else:
            action = controller.decide(proposal, slot, telemetry)
            distance = float(
                np.sqrt(
                    np.dot(env.space.weights(config.contract.weights), (action - proposal) ** 2)
                )
            )
            info = {
                "intervention": distance > config.contract.solver_tolerance,
                "fallback": False,
                "projection_distance": distance,
                "normalized_distance": distance
                / np.sqrt(env.space.weights(config.contract.weights).sum()),
                "supported": None,
                "proposal_risk": None,
                "reason": method,
            }
        y = env.step(action)
        if not method.startswith("pandora"):
            controller.feedback(action, y)
        obs.append(observation)
        proposals.append(proposal)
        actions.append(action)
        outcomes.append(y)
        rows.append(
            {
                "seed": seed,
                "site": site,
                "slot": slot,
                "method": method,
                **dict(
                    zip(
                        [
                            "throughput_mbps",
                            "delay_ms",
                            "loss_ratio",
                            "violation_ratio",
                            "fairness",
                        ],
                        y,
                    )
                ),
                **info,
            }
        )
    episode = Episode(
        episode_id,
        "test",
        np.array(obs),
        np.array(proposals),
        np.array(actions),
        np.array(actions),
        np.array(outcomes),
    )
    metrics = episode_metrics(episode.kpis, config.contract.epsilon)
    for name in ("intervention", "fallback", "projection_distance", "normalized_distance"):
        metrics[name] = float(np.mean([row[name] for row in rows]))
    reliability = None
    if model is not None:
        predictions, probabilities = model.predict(episode.features)
        reliability = prediction_metrics(
            episode.kpis, predictions, probabilities, config.contract.epsilon
        )
        reliability.update(
            {
                "candidate_support_coverage": sum(s["supported"] for s in syntheses)
                / max(1, sum(s["candidates"] for s in syntheses)),
                "verification_failure_period_ratio": float(
                    np.mean([s["verification_failures"] > 0 for s in syntheses])
                ),
                "calibration_margin": calibrator.margin,
                "support_threshold": calibrator.threshold,
            }
        )
    return episode, rows, metrics, reliability, syntheses


def run_experiment(config, output, methods=("independent", "adaptive", "pandora"), progress=print):
    config.validate()
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"Choose unique methods from {METHODS}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty to prevent mixing experiments: {output}")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    write_json(output / "config.json", config.to_dict())
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    manifest = {
        "backend": AnalyticalRAN.backend_name,
        "paper_results_reproduced": False,
        "methods": list(methods),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": source_hash.hexdigest(),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "osqp", "torch", "PyYAML", "matplotlib")
        },
        "test_updates": False,
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    seed_metrics = {method: {} for method in methods}
    all_reliability = {}
    with (output / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for seed in config.seeds:
            progress(f"seed={seed}: collecting disjoint train/calibration episodes")
            datasets = []
            for site in range(config.simulation.sites):
                local = output / f"seed-{seed}" / f"site-{site}"
                episodes = []
                for episode_id in range(config.simulation.train_episodes + 1):
                    split = (
                        "train" if episode_id < config.simulation.train_episodes else "calibration"
                    )
                    episode = collect_episode(config, seed, site, episode_id, split)
                    episode.save(local / f"{split}-{episode_id}.npz")
                    episodes.append(episode)
                datasets.append(episodes)
            training_cache = {}
            for method in methods:
                current = method_config(config, method)
                models, calibrators = None, None
                if method.startswith("pandora"):
                    key = (
                        current.learning.personalize,
                        current.learning.federate,
                        current.learning.calibrate,
                    )
                    if key not in training_cache:
                        progress(f"seed={seed} method={method}: training federated impact models")
                        training_cache[key] = train_models(current, datasets, seed)
                    models, calibrators, logs = training_cache[key]
                    write_json(output / f"training-{seed}-{method}.json", logs)
                    for site, model in enumerate(models):
                        torch.save(
                            {
                                "input_dim": model.input_dim,
                                "state_dict": model.state_dict(),
                                "config": current.to_dict(),
                                "site": site,
                            },
                            output / f"seed-{seed}" / f"site-{site}" / f"{method}.pt",
                        )
                progress(f"seed={seed} method={method}: evaluating frozen models")
                metrics = []
                for site, episodes in enumerate(datasets):
                    episode, rows, metric, reliability, syntheses = evaluate_site(
                        current,
                        method,
                        seed,
                        site,
                        episodes[-1],
                        models[site] if models else None,
                        calibrators[site] if calibrators else None,
                    )
                    local = output / f"seed-{seed}" / f"site-{site}"
                    episode.save(local / f"test-{method}.npz")
                    write_json(local / f"contracts-{method}.json", syntheses)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    metrics.append(metric)
                    all_reliability[f"{seed}/{site}/{method}"] = reliability
                # Site-level episode statistics are averaged before bootstrapping seeds.
                seed_metrics[method][seed] = {
                    k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]
                }
    rng = np.random.default_rng(741)
    summary = {
        method: {
            k: bootstrap_mean([m[k] for m in values.values()], rng, config.bootstrap_samples)
            for k in next(iter(values.values()))
        }
        for method, values in seed_metrics.items()
    }
    differences = {}
    if "pandora" in methods:
        for method in methods:
            if method != "pandora":
                differences[method] = {
                    k: paired_difference(
                        {s: m[k] for s, m in seed_metrics["pandora"].items()},
                        {s: m[k] for s, m in seed_metrics[method].items()},
                        rng,
                        config.bootstrap_samples,
                    )
                    for k in summary[method]
                }
    write_json(
        output / "metrics.json",
        {
            "backend": AnalyticalRAN.backend_name,
            "summary": summary,
            "paired_difference_from_pandora": differences,
            "seed_metrics": seed_metrics,
            "reliability": all_reliability,
        },
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "mean", "ci95_low", "ci95_high", "seeds"])
        for method, metrics in summary.items():
            for key, result in metrics.items():
                ci = result["ci95"] or ["", ""]
                writer.writerow([method, key, result["mean"], *ci, result["seeds"]])
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return summary
