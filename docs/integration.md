# External RAN integration

Pandora's policy boundary is independent of transport. The repository provides a working JSON-lines process interface and a channel importer. Connect E2AP, E2SM, or ns-3 scheduling hooks on the platform side to apply the projected vector and return post-execution measurements.

## Upstream projects

- [O-RAN SC ns-3 E2 module](https://github.com/o-ran-sc/sim-ns3-o-ran-e2): the upstream README describes its ns3-mmWave extension and custom e2sim dependency. Pin mutually compatible commits in your deployment environment rather than assuming a generic ns-3 release works.
- [ns-O-RAN framework](https://openrangym.com/ran-frameworks/ns-o-ran): upstream integration information.
- [Fraunhofer HHI QuaDRiGa](https://github.com/fraunhoferhhi/QuaDRiGa): official channel generator and its own license. Export channel/SINR traces from your actual scenario; record the scenario and trace configuration alongside imported samples.
- [OSQP Python interface](https://osqp.org/docs/interfaces/python.html): the sparse QP interface used by the contract projector.

These dependencies are not downloaded or built by `pip install pandora-oran`. The package name here identifies the local source distribution; install this checkout as documented in the README.

## Control loop

```text
platform observes context o_t and collects proposal a_bar_t
    → send JSON step to pandora control
    ← receive projected action a_tilde_t
platform applies rho/nu/omega with its configured control hooks
platform measures actual a_exec_t and post-action y_(t+1)
    → include that acknowledgement with the next step
```

Use the runnable example first:

```bash
pandora run --config configs/smoke.yaml --output runs/bridge-training
python examples/external_loop.py --run runs/bridge-training --output runs/bridge-audit.jsonl
```

The example starts a separate Python controller process and exchanges one request/reply per slot. Its plant is the Python RAN environment. The final `finish` request acknowledges the last action, so the last transition is not lost.

To launch the policy directly (replace the six slice IDs with the actual site's assignments):

```bash
pandora control --checkpoint runs/bridge-training/seed-0/site-0/pandora.pt --calibration runs/bridge-training/seed-0/site-0/calibration-4.npz --slices 0,0,1,1,2,2 --audit runs/site-0-audit.jsonl
```

`--slices` cannot be guessed from the model dimensions. It defines the site-specific reserve rows and must match the deployment topology.

## Protocol v1

Every line is one JSON object. Slots are zero-based and consecutive. Only identical retries of the most recently accepted request return the cached response; a changed or older retry is rejected. Use one process per site. A new process begins a new episode at slot 0; persistent resume is not implemented.

First request:

```json
{
  "version": 1,
  "type": "step",
  "slot": 0,
  "observation": ["O finite numeric feature values; see layout below"],
  "proposal": ["A finite numeric action values"],
  "feedback": null
}
```

The string entries above describe array lengths; they are schema notation, not a runnable request. The example script generates valid numeric messages.

Response:

```json
{
  "version": 1,
  "slot": 0,
  "action": ["A projected numeric action values"],
  "fallback": true,
  "intervention": true,
  "projection_distance": 0.12,
  "reason": "unsupported-proposal"
}
```

At slot `t > 0`, add the previous slot's acknowledgement:

```json
{
  "slot": 0,
  "executed": ["A actually applied numeric action values"],
  "kpis": [100.0, 12.0, 0.01, 0.05, 0.9]
}
```

KPI order is `[site_total_throughput_Mbps, mean_UE_delay_ms, loss_ratio, UE_violation_ratio, Jain_fairness]`. `feedback.slot` must be `t-1`. Request `{"version":1,"type":"finish","slot":T,"feedback":...}` after the last execution. Errors return a JSON object with `error` and do not advance a validated protocol slot.

The policy validates hard action-domain compliance and numeric dimensions. It records proposal, projection, actual execution and feedback in the local audit. A platform that discretizes or otherwise changes the projected action must report the actual applied vector and separately assess whether its mapping preserves the contract. Do not label this mapping as identity unless verified.

## Feature and action order

The default observation is the concatenation of:

| Block | Length | Scaling |
|---|---:|---|
| Per-UE incoming rate | U | Mbps / 10 |
| Per-UE, per-cell capacity, row-major | U×G | Mbps / 200 |
| Per-UE slice one-hot, row-major | U×3 | Slice IDs 0/1/2 |
| Queue occupancy | U | Mbit / configured buffer Mbit |
| Previous per-UE throughput | U | Mbps / 10 |
| Previous per-UE delay | U | ms / 1000 |
| Cell load | G | Resource occupancy ratio |

Thus `O = U×(G+7)+G`. The action concatenation is `[rho(U), nu(U×G), omega(U)]`, so `A = U×(G+2)`. Model input length is `O+A`. Share the same schema across sites, or change the model and data adapters consistently before training.

The package records all five KPI outcomes. It does not infer real KPI values from projected actions in external-control mode. Training is a separate explicit command using acknowledged transitions converted to `Episode` files.

## QuaDRiGa/SINR import

Export a CSV with a row for every `(slot, ue, cell)`, using zero-based indices:

```csv
slot,ue,cell,sinr_db
0,0,0,12.5
0,0,1,9.2
```

The file must contain the entire configured Cartesian grid; duplicate/missing indices are errors. Convert dB to linear SINR before the Shannon formula; the importer handles this explicitly.

```bash
pandora import-trace data/sinr.csv --slots 3600 --users 20 --cells 2 --bandwidth-mhz 20 --output data/channel.npz
```

The result contains `capacity_mbps[T,U,G]`. The simulation backend accepts it through the Python API:

```python
import numpy as np
from pandora.config import load_config
from pandora.simulator import AnalyticalRAN

config = load_config("configs/paper.yaml")
with np.load("data/channel.npz", allow_pickle=False) as trace:
    ran = AnalyticalRAN(
        config.simulation, seed=0, site=0, episode=0, channel_trace=trace["capacity_mbps"]
    )
```

This substitutes channel capacities only. Queueing, service and KPIs remain analytical and must retain that backend label. A true ns-3 run instead uses platform-side traces and the JSONL control loop, with post-execution KPI collection in ns-3.
