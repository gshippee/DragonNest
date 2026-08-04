# DragonNest

DragonNest combines the best parts of the two hackathon repos in this
workspace:

- `PersonaCare`: a concrete on-device Snapdragon demo pipeline for OCR,
  structured extraction, email drafting, speech, and local QAIRT/QNN execution.
- `PersonaCare-Steering-Research`: tested activation steering, ONNX/QNN
  steering validation, and layer split-compute experiments.

The goal is one unified Snapdragon AI fabric:

> A trusted set of Snapdragon devices can route tasks, run multimodal pipelines,
> steer model behavior, split work across devices, and reroute around thermal,
> battery, memory, and availability constraints.

## What This Repo Is

DragonNest is the umbrella product/repo for:

- **SnapRouter control plane**: device/model routing, health-aware scheduling,
  reroute, data parallelism, layer-pipeline planning, and route explainability.
- **PersonaCare demo app**: doctor-note/photo/audio workflows that prove the
  product value in a user-facing scenario.
- **Steering research path**: activation-steering metadata, policy checks, and
  future QNN/Genie integration.
- **Split-compute path**: pipeline-stage model execution with hidden-state
  boundary tensors.

The first version is intentionally lightweight: it includes a runnable mock
core and clear extension seams for gRPC agents, FastAPI dashboard, QAIRT/QNN,
Genie, and real device execution.

## Current Status

Implemented in this scaffold:

- Rule-based task classifier.
- Steering vector registry and compatibility checks.
- Execution planner for `single`, `data_parallel`, and `layer_pipeline`.
- Health-aware deterministic router.
- Mock executor for single, shard, and pipeline-stage execution.
- Unit tests for classifier, planner, steering policy, and routing.

Not implemented yet:

- Persistent gRPC agent streams.
- FastAPI dashboard.
- Real QAIRT/QNN/Genie execution adapters.
- Real tensor transport for layer-pipeline execution.

## Repository Layout

```text
DragonNest/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- configs/
|   |-- dev-fabric.yaml
|   `-- steering-vectors.yaml
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MIGRATION_PLAN.md
|   `-- SNAPROUTER.md
|-- scripts/
|   `-- demo_mock.py
|-- src/
|   `-- dragon_nest/
|       |-- __init__.py
|       |-- classifier.py
|       |-- executors.py
|       |-- models.py
|       |-- planner.py
|       |-- router.py
|       `-- steering.py
`-- tests/
    |-- test_classifier.py
    |-- test_planner_router.py
    `-- test_steering.py
```

## Quick Start

```bash
cd DragonNest
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/demo_mock.py
```

For editable development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Product Threads

DragonNest keeps three product threads under one roof:

1. **Multimodal care demo**
   - OCR -> structured extraction -> drafted email -> TTS/audio.
   - Based on the practical `PersonaCare` pipeline.

2. **Trusted Snapdragon fabric**
   - Brain + Device Agents.
   - Health-aware routing and reroute.
   - Single-device, data-parallel, and layer-pipeline execution.

3. **Behavior control**
   - Vector steering metadata and policy.
   - Runtime steering where supported.
   - Compiled steering variants where runtime steering is unavailable.

## Source Repo Relationship

This repo should not blindly vendor the two source repos. Use them as follows:

- Port stable low-level runtime wrappers from `PersonaCare` when DragonNest is
  ready for real QAIRT/QNN execution.
- Port tested steering/split-compute modules from `PersonaCare-Steering-Research`
  behind DragonNest interfaces.
- Keep generated artifacts, downloaded model binaries, and hardware-specific
  cache files out of git.

See [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md) for the staged merge plan.

