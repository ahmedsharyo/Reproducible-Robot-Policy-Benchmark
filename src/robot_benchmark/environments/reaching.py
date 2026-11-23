"""Small 2-D reaching task with optional MuJoCo backend."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReachingSpec:
    observation_dim: int = 4  # end-effector xy, goal xy
    action_dim: int = 2       # normalized Cartesian delta xy
    action_scale: float = 0.05
    goal_radius: float = 0.06
    workspace_low: float = -1.0
    workspace_high: float = 1.0


class ReachingEnv:
    """Point reaching, with a NumPy integrator or MuJoCo point mass."""

    def __init__(self, backend: str = "numpy", max_steps: int = 50, spec: ReachingSpec | None = None):
        if backend not in {"numpy", "mujoco"}:
            raise ValueError(f"unsupported backend: {backend}")
        self.backend, self.max_steps = backend, int(max_steps)
        self.spec = spec or ReachingSpec()
        self.rng = np.random.default_rng(0)
        self.steps = 0
        self.ee = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self._mj = None
        if backend == "mujoco":
            try:
                import mujoco
            except ImportError as exc:
                raise RuntimeError("MuJoCo backend requested; install robot-policy-benchmark[mujoco]") from exc
            xml = '''<mujoco model="point_reach"><option timestep="0.01" gravity="0 0 0"/><worldbody>
              <body name="point" pos="0 0 0"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><geom type="sphere" size="0.04" mass="1" rgba="0.1 0.3 0.9 1"/></body>
              <body name="goal" pos="0 0 0"><geom type="sphere" size="0.06" contype="0" conaffinity="0" rgba="0.2 0.8 0.2 0.5"/></body>
            </worldbody><actuator><motor name="mx" joint="x" gear="1"/><motor name="my" joint="y" gear="1"/></actuator></mujoco>'''
            model = mujoco.MjModel.from_xml_string(xml)
            self._mj = (mujoco, model, None)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.ee = self.rng.uniform(-0.55, 0.55, size=2).astype(np.float32)
        self.goal = self.rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
        if options and "ee" in options:
            self.ee = np.asarray(options["ee"], dtype=np.float32).reshape(2)
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32).reshape(2)
        if self._mj:
            mj, model, _ = self._mj
            model.body("goal").pos[:2] = self.goal
            data = mj.MjData(model)
            data.qpos[:] = self.ee
            data.qvel[:] = 0
            self._mj = (mj, model, data)
            mj.mj_forward(model, data)
        return self._obs(), {"success": self._success(), "backend": self.backend}

    def _obs(self):
        if self._mj:
            data = self._mj[2]
            self.ee = np.asarray(data.qpos[:2], dtype=np.float32).copy()
        return np.concatenate((self.ee, self.goal)).astype(np.float32)

    def _success(self):
        return bool(np.linalg.norm(self.ee - self.goal) <= self.spec.goal_radius)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError("action must be a finite vector of shape (2,)")
        a = np.clip(a, -1, 1)
        delta = a * self.spec.action_scale
        self.steps += 1
        if self._mj:
            mj, model, data = self._mj
            # This benchmark's action contract is a Cartesian position delta.
            # Set state directly in MuJoCo to match the NumPy reference integrator;
            # dynamics/control-latency studies should use a separately specified task.
            data.qpos[:] = np.clip(self.ee + delta, self.spec.workspace_low, self.spec.workspace_high)
            data.qvel[:] = delta / 0.01
            mj.mj_forward(model, data)
            self.ee = np.asarray(data.qpos[:2], dtype=np.float32).copy()
        else:
            self.ee = np.clip(self.ee + delta, self.spec.workspace_low, self.spec.workspace_high)
        success = self._success()
        terminated, truncated = success, self.steps >= self.max_steps and not success
        reward = float(success)
        return self._obs(), reward, terminated, truncated, {"success": success, "distance": float(np.linalg.norm(self.ee-self.goal))}

    def close(self):
        pass


def scripted_expert(observation: np.ndarray, action_scale: float = 0.05) -> np.ndarray:
    """Greedy bounded Cartesian controller in normalized action units."""
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (4,):
        raise ValueError("reaching observation must have shape (4,)")
    return np.clip((obs[2:] - obs[:2]) / action_scale, -1.0, 1.0).astype(np.float32)
