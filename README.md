# Reproducible Robot Policy Benchmark

An open-source, local-first framework for controlled comparisons of learned manipulation policies. The initial milestone is a small deterministic planar reaching task with scripted demonstrations and a single-action behavior cloning baseline. It is intended for reproducible reduced-scale experiments on Apple Silicon and CPU.

## Compatibility and scope

- Python 3.11+ is declared; Python 3.12 is available on the development Mac. Install a compatible PyTorch wheel for MPS or use CPU fallback (`--device cpu`). The implementation selects MPS when available.
- MuJoCo and Gymnasium are optional extras. The task has a deterministic NumPy fallback so data generation works without a simulator. The MuJoCo adapter uses MuJoCo state and geometry but directly sets Cartesian position to preserve the declared delta action contract; it does not model actuator dynamics. Install `.[mujoco]` to enable it.
- The initial task is 2-D point reaching, not a grasping or pick-and-place benchmark. The same state and Cartesian-delta action specification is used for every baseline.
- robomimic and LeRobot are optional dependencies only. No dataset, controller, embodiment, image schema, or SmolVLA checkpoint was inspected or validated in this environment; therefore those adapters and closed-loop claims are not enabled. Do not compare their metrics to this task until compatibility is established.
- No experiment results are checked in. Run the commands below to produce actual local outputs.

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mujoco,analysis]'
robot-benchmark run --config configs/experiments/smoke.yaml
```

For simulator-free smoke use, omit `mujoco` and set `environment.backend: numpy` in the config. Outputs are written under `results/<run_id>/` (dataset, checkpoint, metrics, and metadata). To force CPU, pass `--device cpu`.

## Pipeline

The deterministic expert moves a point end effector toward a sampled goal with bounded Cartesian deltas. Demonstrations are saved as compressed NPZ with schema/version metadata and a SHA-256 sidecar. Episodes are split before training windows/examples are formed; normalization is fit only on training episodes. Training seeds and evaluation seeds are distinct and persisted. The CLI runs generation, train/validation, checkpointing, and paired held-out closed-loop evaluation end to end.

The current config is a compact smoke run, not evidence for any research hypothesis. Increase demonstration/evaluation counts and run multiple `training_seeds` for research. Future phases should add chunked BC, a small conditional diffusion policy, diversity/generalization/robustness ablations, and delayed execution with separate explicit protocols.

## Commands

```bash
robot-benchmark run --config configs/experiments/smoke.yaml
pytest -q
```

## Reproducibility and limitations

The runner records package versions, selected backend/device, full config, split episode IDs, and independent evaluation seeds. This foundation does not yet provide confidence intervals, video capture, image policies, statistical factorial analysis, or asynchronous scheduling. It must not be described as satisfying those later phases. Wall-clock timings are local measurements and are not real-time robot execution claims.
