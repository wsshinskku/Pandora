# Validation record

Local validation uses Python 3.12 on Windows 11, CPU PyTorch. Exact package versions are saved in `requirements-tested.txt` and the checked-in [smoke manifest](assets/smoke-manifest.json).

## Automated checks

- Numerical weighted projection with a closed-form optimum.
- OSQP solutions compared against an independent SciPy SLSQP constrained optimizer.
- Projection identity, resource budgets, steering simplexes and full coupled feasibility.
- Nonempty fallback containing neutral, including a singleton fallback.
- Interior contract sampling and nested shrink feasibility.
- Exactly 128 valid candidates and both accepted-contract and high-risk rejection paths.
- Verification rejection across all four checks and three allowed shrinks.
- Leave-one-out support calibration and residual upper-risk clipping.
- Sample-weighted FedAvg and preservation of private site adapters.
- Rejection of calibration/test data in training and duplicate episode IDs.
- Applied-action supervision, checkpoint serialization and NPZ round trip.
- Paired exogenous traces, tail direction, bootstrap pairing and single-class AUROC handling.
- Missing/duplicate channel traces and invalid configuration rejection.
- JSON scientific-notation configuration reload and small QP residuals in adaptive bounds.
- Complete paired-seed experiment plus external acknowledgement/retry protocol.

The current suite contains 21 passing tests. `ruff check` and `ruff format --check` pass. Both a source archive and a wheel build successfully.

## Executed scenarios

```bash
pandora run --config configs/smoke.yaml --output runs/smoke-final --methods independent static adaptive scheduler-reference qos-reference pandora
pandora ablate --config configs/smoke.yaml --output runs/ablation
pandora sweep --config configs/smoke.yaml --output runs/sweep-verified --values 0.5 1.0 1.5 --seeds 0 1
python examples/external_loop.py --run runs/smoke --output runs/external-audit.jsonl
pandora train --config configs/smoke.yaml --data runs/smoke/seed-0 --output runs/offline-training
```

The external-process example records 90 acknowledged transitions, including the final slot. The sweep executes six paired seed/aggressiveness combinations and writes seed-level intervals. All four Pandora ablations execute.

The paper-scale simulation configuration also completes for seed 0:

```bash
pandora run --config configs/paper.yaml --output runs/paper-one-seed --seeds 0
```

This runs four sites with 20 UEs and two cells each, four 3600-slot training episodes, one calibration episode and one test episode per site. All 24 federated rounds finish. Independent, adaptive and Pandora evaluations each execute 14,400 test control slots across the sites. The [summary](assets/paper-scale-one-seed-summary.csv), [configuration](assets/paper-scale-one-seed-config.json) and [manifest](assets/paper-scale-one-seed-manifest.json) record the actual run.

At this scale the seed-0 mean throughputs are 41.2325 Mbps (independent), 41.1206 Mbps (adaptive), and 41.2129 Mbps (Pandora). Pandora uses fallback in every slot. Seed-level confidence intervals require multiple independent seeds.

## GitHub Actions

[Validation run 35524349260](https://github.com/wsshinskku/Pandora/actions/runs/35524349260) passes all three environments for implementation commit `f0332dc`:

| Environment | Result |
|---|---|
| Ubuntu, Python 3.11 | Passed |
| Ubuntu, Python 3.12 | Passed |
| Windows, Python 3.12 | Passed |

Each job checks lint/formatting, runs the test suite, executes the small experiment and external-process example, builds the package, and uploads its generated smoke summary. Subsequent documentation-only changes record these results without modifying the implementation.

## Observed small-run behavior

The checked smoke configuration uses two sites, six UEs/site, 90 slots, four training episodes and seed 0. Its results include:

| Method | Mean site throughput (Mbps) | Mean delay (ms) | Exceedance probability | Fallback ratio |
|---|---:|---:|---:|---:|
| Independent | 10.8186 | 5.0168 | 0.0556 | N/A |
| Adaptive | 10.8186 | 5.0080 | 0.0556 | N/A |
| Pandora | 10.8186 | 5.0000 | 0.0556 | 1.0000 |

The small load leaves throughput largely arrival-limited. Pandora uses fallback for all slots with this calibration/training sample. A dedicated numerical test also exercises successful learned-contract synthesis and verification.

The [figure](assets/smoke-performance.png), [flat metrics](assets/smoke-summary.csv) and [resolved configuration](assets/smoke-config.json) come from the actual run. Only generated summaries and the figure are checked in; raw local episodes and model checkpoints stay under the ignored `runs/` directory.

## Validation boundaries

Local checks cover the Python entry point and controller subprocess protocol. GitHub Actions covers Linux and Windows as recorded above. Docker build and live external RAN integration are separate validation targets.