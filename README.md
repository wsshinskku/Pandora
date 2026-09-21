# Pandora

**Personalized Adaptive Network-Driven Open RAN Orchestration with Federated Contracts**

[![Tests](https://github.com/wsshinskku/Pandora/actions/workflows/ci.yml/badge.svg)](https://github.com/wsshinskku/Pandora/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)

[한국어 설명](README.ko.md) · [Paper-to-code mapping](docs/paper-to-code.md) · [Reproducibility](docs/reproducibility.md) · [External RAN integration](docs/integration.md)

Pandora coordinates resource-allocation, traffic-steering, and QoS xApps without replacing their policies. A site learns the effects of joint actions, constructs a locally calibrated feasible region, and makes the smallest weighted correction to proposals outside that region. Sites share model updates while keeping their transitions, adapters, calibration residuals, and support statistics local.

This is the **official implementation of Pandora**, maintained by the paper’s author. It includes personalized federated learning, contract synthesis, runtime projection, a Python RAN simulation environment, experiment tools, and external RAN integration interfaces.

## Quick start

Requirements: Python 3.11 or newer and a CPU. CUDA is not required. Run commands from the repository root.

```bash
git clone https://github.com/wsshinskku/Pandora.git
cd Pandora
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install and execute the complete small experiment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
pandora run --config configs/smoke.yaml --output runs/first
```

The command collects separate training/calibration episodes, trains site models with FedAvg, calibrates risk, runs frozen-model test episodes, writes metrics and checkpoints, and renders a figure. Choose a **new output directory** for each run; the CLI refuses to mix or overwrite experiments.

For a CPU-only PyTorch installation, install PyTorch from its CPU index before installing this package:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## What is implemented

| Component | Implementation |
|---|---|
| Joint xApp actions | Resource shares `rho[U]`, per-UE steering simplex `nu[U,G]`, priorities `omega[U]` |
| Action-impact model | Shared `128 → 128 → 64` ReLU MLP, five-KPI regression head, binary risk head |
| Personalization | Site-private affine residual adapter on the 64-dimensional representation |
| Federated learning | Sample-weighted FedAvg of shared trunk/heads; adapters stay local |
| Support checking | Standardized nearest calibration sample; 95% leave-one-out distance quantile |
| Risk calibration | One-sided residual quantile and clipped upper-risk score |
| Candidates | Current proposal + 15 recent + 64 perturbations + 48 domain samples |
| Contract synthesis | Quantile box, sparse coupled inequalities, 64 verification actions, up to 3 shrinks |
| Runtime enforcement | Per-slot support guard and weighted OSQP projection, with a nonempty fallback |
| Evaluation | Held-out KPIs, tails, SLA exceedance, interventions, prediction/reliability metrics |
| Statistics | Seed-level bootstrap intervals and paired differences using shared seed IDs |
| Integration | Checkpoints, strict JSONL proposal/acknowledgement protocol, SINR CSV importer |

The adapter, coupling constraints, traffic model, SLA thresholds, and exploration settings are documented in the [paper-to-code mapping](docs/paper-to-code.md).

## Architecture

```mermaid
flowchart LR
    RAN["RAN telemetry"] --> X["Independent black-box xApps"]
    X --> P["Joint proposal: rho, nu, omega"]
    RAN --> M["Shared impact model + local adapter"]
    M --> S["128 candidates → support and risk screen"]
    S --> C["Quantile contract → verify / shrink / fallback"]
    P --> Q["Runtime support guard + weighted QP"]
    C --> Q
    Q --> E["Applied RAN action"]
    E --> D["Site-local transitions and KPI feedback"]
    D --> M
    M -->|"shared weights + sample count"| F["FedAvg server"]
    F -->|"shared weights"| M
```

For proposal `a_bar`, Pandora solves

```text
minimize    (a - a_bar)^T Lambda (a - a_bar)
subject to  lower <= a <= upper
            H a <= h
            sum_g nu[u,g] = 1   for each UE u
            sum_u rho[u] <= number_of_cells
```

The block weights are `(1.0, 1.5, 2.0)`. Larger weights preserve that xApp's proposal more strongly. Feasible proposals pass through unchanged within the solver tolerance. Executed decisions are not clipped or renormalized after projection because doing so could break coupled constraints.

## Experiments

### Compare methods

```bash
pandora run --config configs/smoke.yaml --output runs/comparison --methods independent static adaptive scheduler-reference qos-reference pandora
```

| CLI method | Meaning |
|---|---|
| `independent` | Execute the three xApps' concatenated proposals |
| `static` | Project into a fixed calibration-derived fallback contract |
| `adaptive` | Update a local 10%–90% box from compliant applied actions |
| `scheduler-reference` | Round-robin owner selection; suppress other blocks to neutral |
| `qos-reference` | Queue-based resource and priority correction heuristic |
| `pandora` | Full personalized, calibrated, coupled contract pipeline |

`scheduler-reference` selects one xApp at a time; `qos-reference` applies queue-based corrections. The table above lists the six available comparison methods. See the [method mapping](docs/paper-to-code.md) for algorithm details.

### Ablation study

```bash
pandora ablate --config configs/smoke.yaml --output runs/ablations
```

Runs `pandora`, `pandora-no-personalization`, `pandora-no-fl`, `pandora-no-calibration`, and `pandora-box-only`. Box-only removes learned coupling rows while retaining the hard domain, steering simplexes and aggregate resource budget.

### Aggressiveness and paired seeds

```bash
pandora sweep --config configs/smoke.yaml --output runs/aggressiveness --values 0.5 1.0 1.5 --seeds 0 1 2
pandora run --config configs/paper.yaml --output runs/paper-scale --seeds 0 1 2
```

`paper.yaml` records the manuscript's 4 sites, 20 UEs/site, 2 cells/site, 3600 slots, 30-slot contract periods, 600-slot FL periods, and 4/1/1 episode split. Its default seed list contains 20 seeds. Runtime depends on hardware and how often synthesis reaches verification.

### Outputs

```text
runs/first/
├── config.json                  # Fully resolved configuration
├── manifest.json                # Backend, source hash, package versions, completion status
├── telemetry.csv                # Per-slot KPIs and contract decisions
├── metrics.json                 # Means, CIs, paired differences, prediction reliability
├── summary.csv                  # Flat table for analysis
├── performance.png              # Throughput CDF, delay CCDF, intervention ratio
├── training-<seed>-<method>.json # Losses, calibration margins, payload sizes
└── seed-<seed>/site-<site>/
    ├── train-<episode>.npz
    ├── calibration-<episode>.npz
    ├── <method>.pt               # Shared model and this site's private adapter
    ├── test-<method>.npz
    └── contracts-<method>.json
```

All ratios are stored in `[0,1]`, throughput in Mbps, delay in milliseconds, and queues in Mbit. Top-10% delay means the **lowest** delay samples; bottom-10% delay means the **highest**. Metrics are computed per site/episode, averaged within each seed, then bootstrapped over seeds. One-seed runs have `ci95: null`, since there is no seed-level uncertainty estimate.

![Measured smoke-run results](docs/assets/smoke-performance.png)

The figure shows the included smoke experiment. Its contract acceptance and fallback rates are recorded alongside throughput, delay, and interventions; see the [validation record](docs/validation.md).

## Connect an external RAN process

```bash
python examples/external_loop.py --run runs/first --output runs/external-audit.jsonl
```

The example connects a separate controller process to the Python RAN environment through JSONL. Connect your ns-O-RAN telemetry and control hooks at the same interface. `pandora control` emits projected actions and requires acknowledgement of the applied action before the next slot; both are retained for action-impact learning.

[Integration instructions](docs/integration.md) cover the JSON schema, feature order, channel import, upstream dependencies, and the boundary between this repository and the platform-specific E2/scheduler implementation.

To train from external site-local episodes in the same NPZ schema:

```bash
pandora train --config configs/paper.yaml --data data --output runs/external-models --seed 0
```

See the [dataset layout and training protocol](docs/reproducibility.md#external-episode-training).

## Repository layout

```text
src/pandora/
  actions.py       # Action domain and coupling constraints
  contracts.py     # QP, fallback, feasibility, hit-and-run sampling
  calibration.py   # Support distances and empirical residual margin
  learning.py      # Shared MLP, local adapter, FedAvg
  synthesis.py     # Algorithm 1
  controller.py    # Slot-level orchestration
  simulator.py     # RAN traffic, channel and queue simulation
  xapps.py         # Independent black-box controllers
  baselines.py     # Runnable comparison heuristics
  experiment.py    # Train/calibrate/evaluate/export pipeline
  metrics.py       # Prediction, tails, paired bootstrap
  integration.py  # External-process control protocol
  traces.py        # Strict SINR CSV import
  plotting.py      # Figures from actual run data
  cli.py           # User-facing commands
configs/           # Small and manuscript-scale settings
examples/          # Executable external-process example
tests/             # Numerical oracle, data-isolation and integration tests
docs/              # Method mapping, assumptions, integration and validation
```

## Development

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

CI runs these checks and a small end-to-end experiment. The included `Dockerfile` packages the Python simulation and controller.

```bash
docker build -t pandora .
docker run --rm -v "${PWD}/runs:/app/runs" pandora run --config configs/smoke.yaml --output runs/docker
```

## Calibration and fallback

Risk calibration uses held-out residuals, followed by finite candidate verification. Fallback preserves a feasible neutral action. Contract acceptance, empirical SLA outcomes, support coverage, and fallback frequency are recorded separately.

## Citation and license

Associated paper: *Pandora: Personalized Adaptive Network-Driven Open RAN Orchestration with Federated Contracts*. Use [CITATION.cff](CITATION.cff) to cite the software and include the Git revision used in your experiments.

Original repository code is provided under the [MIT License](LICENSE). External tools such as ns-O-RAN and QuaDRiGa retain their own licenses and are not vendored here.
