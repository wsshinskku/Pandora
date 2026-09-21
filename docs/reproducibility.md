# Reproducible experiments

## Execution levels

| Configuration | Purpose | What it establishes |
|---|---|---|
| `configs/smoke.yaml` | Small CPU run, 2 sites × 6 UEs × 90 slots | The complete software pipeline executes |
| `configs/paper.yaml` | Paper-scale topology, time scales and algorithm parameters | Scaling the Python RAN experiment |
| External NPZ episodes + `pandora train` | Learn on separately acquired post-execution data | Applicability to the provided dataset, subject to its provenance |
| External RAN + `pandora control` | Apply contracts to a platform's actual control surface | Requires separately validated telemetry and enforcement hooks |

## Dataset partitioning

For a seed, episodes 0–3 are training, episode 4 is calibration, and episode 5 is testing under the default configuration. The number of training episodes is configurable; the two subsequent episode IDs remain distinct. `partition_episodes` rejects duplicate IDs. `LocalLearner.fit` accepts only training episodes. `RiskCalibrator.fit` accepts only calibration episodes.

Only actually executed actions are model inputs. The simulation backend applies the projected vector unchanged. The external integration additionally records acknowledgements, preserving any difference between requested and executed vectors.

Collection and evaluation use separate random streams. Traffic and channels are generated before actions are executed, so random policy sampling cannot change a method's exogenous trace. A common initial shared model is copied to every site; local shuffling is seeded independently. Each ablation starts from the same initialization and collection data for that seed.

Test labels never optimize neural parameters or calibrate risk. The adaptive baseline is explicitly an online, nonparametric controller and can update its box using its past test-episode feedback. Pandora refreshes contracts from current observations/proposal history, with frozen test-time model parameters.

## Metrics and uncertainty

1. Compute throughput, delay, loss, fairness and SLA outcomes for each post-action slot.
2. Compute mean and tail metrics within each site test episode.
3. Average site metrics to obtain one observation per seed.
4. Resample seed-level observations with replacement 10,000 times in the paper-scale configuration.
5. For comparisons, subtract the Pandora result within each matching seed first, then bootstrap those differences.

Intervals are percentile 2.5%–97.5% intervals; they may be asymmetric. A single seed reports no interval. Small seed counts are only a smoke test of the statistics code. Risk AUROC is `null` when test labels contain only one class. The ECE uses 10 equal-width probability bins, including probability 1 in the last bin.

The `reliability` section of `metrics.json` contains KPI MAE in physical units, raw-risk AUROC/Brier/ECE, support coverage, the fraction of synthesis periods experiencing any verification failure, and local calibration values. Fallback and intervention ratios are in the main summary. Raw probabilities are assessed separately from upper-risk screening scores.

## Example commands

```bash
pandora validate --config configs/paper.yaml
pandora run --config configs/smoke.yaml --output runs/smoke --seeds 0 1 2
pandora ablate --config configs/smoke.yaml --output runs/ablation --seeds 0 1 2
pandora sweep --config configs/smoke.yaml --output runs/sweep --values 0.5 1.0 1.5 --seeds 0 1 2
pandora plot runs/smoke
```

Use separate output roots to compare software revisions. Each manifest records the Python source SHA-256, dependency versions, backend label and completion status. A failed run can leave an incomplete directory marked `running`; inspect it and choose a new output directory. There is no automatic resume or mixed-run append behavior.

`requirements-tested.txt` records the resolved packages in the local validation environment. It includes a CPU PyTorch index. `pyproject.toml` defines the supported dependency ranges for normal installs. Cross-platform floating-point results need not be bitwise identical even when seeds match.

## External episode training

Use the `Episode` schema to write files:

```python
import numpy as np
from pandora.data import Episode

# Arrays obtained from your RAN: observations[N,O], actions[N,A], outcomes[N,5].
# The three action arrays may differ; do not overwrite executed with proposed.
episode = Episode(
    episode_id=0,
    split="train",
    observations=np.asarray(observations),
    proposals=np.asarray(proposals),
    projected=np.asarray(projected),
    executed=np.asarray(executed),
    kpis=np.asarray(outcomes),
)
episode.save("data/site-0/train-0.npz")
```

This snippet defines the schema; the array variables must come from your telemetry collector. Arrange all sites under the same root, with configured training-episode counts and one calibration episode per site. The synchronous runner requires equal total training horizons across sites.

```bash
pandora train --config configs/paper.yaml --data data --output runs/external-models --seed 0
```

Files named `test-*.npz` are not loaded by this command. Episode metadata is still validated. Keep site IDs, UE order, slice assignments, action semantics and feature normalization identical across collection and deployment. The shared model assumes compatible feature/action dimensions across sites.

## Reading an unpromising result

The empirical margin `max(0, quantile(z-p))` can be large if high-risk outcomes are underpredicted. With a 0.10 risk budget, the safe candidate set may then be empty. The correct outcome is fallback, not suppressing the margin to obtain favorable plots. Inspect calibration class balance, model error, support distances, data coverage and the deployment distribution. Report ablations separately; do not silently relax thresholds.
