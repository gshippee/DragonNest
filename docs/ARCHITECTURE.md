# DragonNest Architecture

DragonNest has one control plane and several execution backends.

## Control Plane

The control plane is the future SnapRouter Brain:

- Device registry.
- Health-state cache.
- Task classifier.
- Execution planner.
- Deterministic router.
- Dispatch/retry manager.
- Result reducer.
- Steering vector registry.
- Dashboard API.

The scaffold implements these as local Python modules first. The same models
can later be exposed through gRPC and FastAPI.

## Execution Modes

### Single Device

One task goes to one device/model. This is the default path and the simplest
demo mode.

### Data Parallel

The planner splits a compound task into independent shards. Shards can run on
different devices, then a reducer combines outputs.

Initial reducers:

- `concat`
- `first_success`
- `mock_synthesis`

Future reducers can call a real LLM on the best eligible device.

### Layer Pipeline

A single model is split into layer ranges. Each device advertises model segment
capabilities. A valid pipeline must cover contiguous layers without gaps.

This is based on the local split-compute research pattern:

```text
Part A: input_ids -> hidden boundary
Part B: hidden boundary -> logits/output
```

The MVP uses mock boundary tensors. A future QNN executor can move real hidden
state tensors through the Brain or a trusted transport channel.

### Vector Steering

Steering metadata is first-class in routing and execution. The policy checks:

- vector ID
- model family
- target layer
- alpha range
- positions mode
- local/remote vector sharing

The eventual runtime operation is:

```text
steered_hidden = hidden + alpha * mask * steering_vector
```

The current scaffold passes steering metadata through the route and mock
executor. Real vector loading and QNN/Genie integration come later.

## Source Repo Inheritance

From `PersonaCare`:

- QAIRT/QNN runner pattern.
- EasyOCR, MeloTTS, Whisper, and Genie orchestration lessons.
- Doctor-note end-to-end demo.
- Chunking and local/offline execution practices.

From `PersonaCare-Steering-Research`:

- Tested steering injection.
- ONNX export and backend comparison.
- QNN/AI Hub proof workflow.
- Layer split-compute proof.
- Reproducibility and result artifact discipline.

