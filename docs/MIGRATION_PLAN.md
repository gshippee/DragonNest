# Migration Plan

DragonNest should merge the two source repos in stages.

## Stage 1: Unify Interfaces

- Keep this scaffold as the top-level repo.
- Define stable interfaces for classifiers, planners, routers, executors,
  reducers, steering registry, and device capabilities.
- Keep mock execution green in tests.

## Stage 2: Bring In Steering

Port from `PersonaCare-Steering-Research`:

- `steering_poc/inject.py`
- `steering_poc/export_onnx.py`
- `steering_poc/quantization.py`
- selected tests for zero-alpha identity and ONNX equivalence

Target DragonNest modules:

- `src/dragon_nest/steering/runtime.py`
- `src/dragon_nest/steering/export_onnx.py`
- `tests/test_steering_runtime.py`

## Stage 3: Bring In Split Compute

Port from `PersonaCare-Steering-Research/src/split_compute`:

- split model wrappers
- local verification
- AI Hub submission flow as optional tooling

Target DragonNest modules:

- `src/dragon_nest/pipeline_split/`
- `src/dragon_nest/executors/qnn_pipeline.py`

## Stage 4: Bring In Multimodal PersonaCare Pipeline

Port from `PersonaCare`:

- `qnn_runner.py`
- `chunking.py`
- `genie_runner.py`
- `easyocr_pipeline.py`
- `melotts_pipeline.py`
- `whisper_pipeline.py`
- `doctor_note_pipeline.py`

Target DragonNest modules:

- `src/dragon_nest/runtime/qnn_runner.py`
- `src/dragon_nest/runtime/genie_runner.py`
- `src/dragon_nest/apps/persona_care/`

Keep hardware-specific paths and downloaded model artifacts in configuration,
not source.

## Stage 5: Add Brain and Agents

- Add `proto/dragonnest.proto`.
- Add Brain gRPC server.
- Add Device Agent client.
- Add FastAPI dashboard.
- Wire the existing local planner/router into the service layer.

## Stage 6: Demo Polish

Required demos:

- Normal route.
- Thermal reroute.
- Private mode.
- Data-parallel fanout.
- Layer-pipeline mock split.
- Vector steering.
- PersonaCare doctor-note workflow.

