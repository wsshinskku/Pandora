# Manuscript-to-code map

Source: the supplied 10-page anonymous manuscript, *Pandora: Personalized Adaptive Network-Driven Open RAN Orchestration with Federated Contracts*.

Source PDF SHA-256: `a3efb0b8cc657e125bf314784901657e29a394d2920534a14b4181ba084efb5a`.

The filename contains “INFOCOM”; that alone does not establish publication or acceptance. This repository is a new implementation, not an imported original author artifact.

## Implemented equations

| Manuscript | Code | Notes |
|---|---|---|
| (1), three time scales | `experiment.py`, `controller.py` | Zero-based code slots: synthesize at `t % K == 0`; training chunks of H transitions |
| (2)–(5), observations and joint proposals | `simulator.Telemetry`, `xapps.ReferenceXApps`, `actions.ActionSpace` | Fixed feature order and per-UE priorities are implementation choices |
| (6)–(10), contract and projection | `contracts.Contract` | Hard domain plus box and coupled inequalities; identity for feasible proposals |
| (11)–(13), network semantics | `simulator.AnalyticalRAN` | Only a sandbox; QoS-dependent sharing and bounded queues are added explicitly |
| (14)–(17), conceptual objective | `synthesis.py`, `learning.py` | Implements Algorithm 1's surrogate, not direct optimization of the entire objective |
| (18)–(20), applied actions and transitions | `data.Episode`, `integration.JsonlSession` | Proposal, projection and actual execution are stored separately |
| (21)–(24), action-impact learning | `learning.ImpactModel`, `LocalLearner` | Five normalized continuous outputs and risk BCE, plus private-adapter L2 penalty |
| (25)–(27), support-aware evaluation | `calibration.RiskCalibrator` | Local standardization and leave-one-out nearest-neighbor distances |
| (28)–(31), empirical risk screen | `RiskCalibrator.fit`, `upper_risk` | Quantile uses NumPy's `higher` order statistic; probabilities clipped to [0,1] |
| (32)–(34), quantile contract fitting | `synthesis.ContractSynthesizer` | H fixed by the reference operator template; right-hand side is adapted |
| (35), region shrinking | `Contract.shrink` | Convex interpolation toward a feasible screened anchor |
| Algorithm 1 | `ContractSynthesizer.synthesize` | Candidate screen, minimum count, verification, shrink, fallback |
| (36)–(39), runtime guard and distance | `controller.PandoraController` | Unsupported proposal uses fallback even if it geometrically fits the learned box |
| (40), FedAvg | `learning.federated_average` | Shared tensors and sample counts only; local adapters never aggregated |
| (41), xApp aggressiveness | `xapps.ReferenceXApps.propose` | Rescale about neutral, then restore the hard action domain |
| (42), SLA exceedance probability | `metrics.episode_metrics` | Strict event `violation_ratio > epsilon` |
| Section V/VI, statistics | `metrics.py`, `experiment.py` | Per-site episode metrics, site average per seed, paired seed bootstrap |

## Values directly supplied by the manuscript

- 4 sites; 20 UEs and 2 cells per site; 3600 one-second slots per episode.
- Contracts every 30 slots; FL every 600 slots.
- Four training, one calibration and one test episode per seed; 20 seeds.
- Shared MLP layers 128, 128, 64; ReLU; Adam at 0.001; batch 256; three local epochs.
- 128 candidates: 1 current, 15 recent, 64 perturbations, 48 domain samples.
- `epsilon = kappa = .10`; `delta_cal = .05`; support quantile .95.
- Box quantiles .10/.90; coupling quantile .90; minimum safe count 24.
- 64 verification samples; shrink factor .20; maximum 3 shrinks.
- Fallback box .20/.80 and coupling quantile .80; fallback must contain neutral action.
- Projection block weights 1.0, 1.5, 2.0; numerical tolerance 1e-5.
- 10,000 seed-level bootstrap resamples for the paper-scale configuration.

## Explicit implementation choices

### Action domain and coupling template

The paper gives the roles of H but not its numerical rows. The reference action domain uses `rho ∈ [0,1]^U`, `nu ∈ [0,1]^(U×G)` with per-UE sum 1, `omega ∈ [0,1]^U`, and `sum(rho) <= G`.

`actions.ActionSpace.coupling_template` builds:

```text
resource / steering: nu[u,g]/G - (U/G)*rho[u] <= 0.20
resource / QoS:      omega[u]  - (U/G)*rho[u] <= 0.35
slice reserve:     -sum(rho[u] for u in slice s) <= -0.10*G*|s|/U
```

These are reference operator constraints, not asserted physical laws or the authors' original H. Neutral is `rho=0.8G/U`, `nu=1/G`, `omega=.5`. The reserve and coupling limits contain that neutral action. Production deployments should supply and validate their own H, limits and neutral policy. Dynamic steering-change rows are not supplied by this reference template.

Fallback quantiles are expanded just enough to include neutral, then intersected with the operator limits. Incompatible limits are rejected. Empty calibration data produces a singleton neutral contract. The fallback's geometric nonemptiness is not an SLA-risk guarantee.

### Learning and calibration

The manuscript leaves the private adapter architecture unspecified. Here it is an affine scale/bias residual on the 64-dimensional trunk output. Trunk and both output heads are shared. Five target scales are fixed at `[100 Mbps, 100 ms, 1, 1, 1]`; the regression loss sums squared normalized errors per sample. No test-fitted scaling is used.

The loss weights are `eta_risk=1` and `eta_loc=1e-4`. Adam is reinitialized at each local update; a site-private deterministic shuffle generator persists. Training uses chronological H-slot blocks of pre-collected training episodes, one block per FL round, rather than silently fitting calibration/test samples. At each broadcast the calibration margin and support model are refreshed. Test episodes freeze neural weights and risk calibration; runtime contracts still refresh every K slots.

Standard-deviation floors are 1e-3 for support coordinates. This avoids division by zero on constant channels or slices. The leave-one-out support threshold uses k=2 nearest neighbors, so the self-distance is excluded. The same calibration partition supplies support statistics and residuals, as specified in the manuscript.

### Candidate verification

Gaussian perturbations use standard deviation .05 before hard-domain normalization. Domain samples use Dirichlet steering and resource proportions. Early slots repeat available history to preserve the specified candidate count; repeated candidates are not statistically independent evidence.

Verification uses hit-and-run samples in the affine hull, initialized with a feasible point and an LP Chebyshev center. There are 32 burn-in moves and 3 moves per retained sample. Samples can be correlated; the procedure is a finite numerical screen, not exhaustive verification.

The safe-set medoid uses Euclidean action distance. Quantile intersections need not contain the unconstrained medoid. If necessary, the implementation chooses the feasible screened candidate minimizing distance to the entire safe set. If no screened candidate is feasible, it falls back. This conservative edge-case resolution is additional to the manuscript's pseudocode. Shrinking toward a feasible anchor preserves nested regions.

QP absolute/relative tolerances are internally set to one tenth of the requested action-feasibility tolerance. Every returned action is checked against the domain and full contract. Solver failure activates fallback; failure of the fallback QP uses its checked neutral action and records the reason. Applied actions are never normalized after the QP.

### Sandbox and experimental protocol

Traffic is gamma-distributed with pre-generated bursts. Channels are correlated synthetic SINR samples converted via `B log2(1+SINR)`. UE slice assignment is stable across episodes for a site, and the four traffic profiles change slice proportions. The sandbox does not claim 3GPP mobility or spatially consistent QuaDRiGa propagation.

Reference values absent from the manuscript include 20 MHz bandwidth, 5 Mbit queues, 5 ms path delay, slice minimum rates `[1,.15,.02]` Mbps and delay limits `[150,50,500]` ms. QoS priorities multiply resource service weights by `.5+omega`; cell overload is shared within the plant. These plant dynamics are intentionally exposed in `simulator.py`.

Training/calibration collection uses 15% independent domain exploration and otherwise perturbed xApp proposals. This collection policy is shared across methods within a seed. Held-out tests use identical pre-generated channel/traffic seeds but evolve their own queues under each controller. The original paper does not provide its data-collection policy or warm-start timeline, so this is an explicit offline protocol choice.

The normalization of reported projection distance is `D/sqrt(sum(weights))`, the weighted RMS distance across unit-range coordinates. Raw distance is also retained. The manuscript does not define its normalized-distance convention.

## Deliberately not claimed

- Reproduction of the original ns-3 measurements, reported percentages, runtime overheads, or 5.2 MB payload size.
- An E2AP transport, E2SM-KPM/RC encoder, ns-3 scheduler patch, RIC container stack, or original QuaDRiGa scenario.
- Original implementations of Sched-xApp, QACM, FRL-Slicing or SafeSlice. Heuristic references have distinct names; the latter two are not included.
- Direct optimization of the utility/change-penalty objective (14); Algorithm 1 does not provide an update rule for those coefficients.
- The independent-trajectory 30-slot risk-ranking experiment in Figure 2(c). The per-slot logs support further analyses, but the checked-in plots do not claim to reproduce this panel.
- Distribution-free or causal safety guarantees, differential privacy, secure aggregation, or a distributed networking service. Federated orchestration is executed synchronously in one process, with an explicit model-only server interface.

The manuscript calls the non-learning comparator “Adaptive Contract” in Table I and surrounding text, but “Static Contract” in Table II. This repository exposes both `adaptive` and `static` explicitly instead of silently conflating them.
