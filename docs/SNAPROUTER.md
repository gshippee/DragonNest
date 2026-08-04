# SnapRouter Track

SnapRouter is the DragonNest control-plane track.

The Brain receives a request, classifies it, builds an execution plan, routes
the plan across trusted Snapdragon devices, dispatches execution, reduces
results, and reroutes around health constraints.

## MVP Brain Components

- Device Registry
- Health Store
- Rule-Based Classifier
- Execution Planner
- Deterministic Router
- Parallel Dispatcher
- Result Reducer
- Steering Registry
- Event Log
- Dashboard API

## MVP Execution Modes

- `single`
- `data_parallel`
- `layer_pipeline`

## MVP Policy Rules

- Healthy and degraded devices are eligible for routing; stale devices are
  last-resort fallbacks, and offline devices cannot receive new work.
- Private mode excludes remote devices.
- Steering requests must match model family, vector ID, layer, alpha range, and
  positions mode.
- Layer pipelines must have contiguous compatible segments.
- Data-parallel shards retry once when an eligible fallback exists.

## Roadmap

1. Keep local mock planner/router tests green.
2. Validate live QNN split boundaries on target hardware.
3. Add production mTLS enrollment.
