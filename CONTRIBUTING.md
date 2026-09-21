# Contributing

Install the editable development package and run `ruff check`, `ruff format --check`, and `pytest` before proposing changes. Include a focused numerical or behavioral test for changes to contract feasibility, model aggregation, calibration, or action execution.

Keep the hard action domain separate from learned constraints. Do not clip a projected action after the QP. Retain the distinction between proposed, projected and executed actions. Keep episode partitions disjoint and shared model updates free of local adapters or raw transitions.

For an experiment, provide the resolved configuration, seed IDs, backend label, manifest, and generated measurements. Document each comparison method’s algorithm and complete configuration.

Do not commit private RAN traces, model checkpoints or credentials. Synthetic result summaries may be placed in `docs/assets` with their manifest and execution command. External platform integration should document compatible upstream commits and feature/action units.
