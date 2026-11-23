"""Canonical episode representation and integrity-checked NPZ storage."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from robot_benchmark.environments import ReachingEnv, ReachingSpec, scripted_expert

SCHEMA_VERSION = 1

@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success: bool
    timestamps: np.ndarray
    metadata: dict

    def validate(self):
        n = len(self.actions)
        if self.observations.shape != (n + 1, 4) or self.actions.shape != (n, 2):
            raise ValueError("episode observation/action boundary or dimensions invalid")
        if self.rewards.shape != (n,) or self.timestamps.shape != (n + 1,):
            raise ValueError("episode rewards/timestamps length invalid")
        if not all(np.all(np.isfinite(x)) for x in (self.observations,self.actions,self.rewards,self.timestamps)):
            raise ValueError("episode contains non-finite values")

@dataclass
class Dataset:
    episodes: list[Episode]
    metadata: dict

    def validate(self):
        if not self.episodes:
            raise ValueError("dataset must contain episodes")
        for ep in self.episodes:
            ep.validate()

def generate_dataset(episodes: int, seed: int, backend: str = "numpy", max_steps: int = 50, action_scale: float = .05, goal_radius: float = .06) -> Dataset:
    env = ReachingEnv(backend, max_steps, ReachingSpec(action_scale=action_scale, goal_radius=goal_radius))
    result=[]
    for i in range(episodes):
        obs,_=env.reset(seed=seed+i)
        obs_list=[obs.copy()]; acts=[]; rewards=[]
        for t in range(max_steps):
            action=scripted_expert(obs, action_scale)
            nxt,r,term,trunc,info=env.step(action)
            acts.append(action); rewards.append(r); obs_list.append(nxt.copy()); obs=nxt
            if term or trunc: break
        ep=Episode(np.asarray(obs_list,dtype=np.float32),np.asarray(acts,dtype=np.float32),np.asarray(rewards,dtype=np.float32),bool(info["success"]),np.arange(len(obs_list),dtype=np.float32)*.01,{"episode_id":i,"seed":seed+i})
        ep.validate(); result.append(ep)
    env.close()
    ds=Dataset(result,{"schema_version":SCHEMA_VERSION,"robot":"point_mass_2d","environment":"reaching-v1","backend":backend,"observation":{"keys":["ee_x","ee_y","goal_x","goal_y"],"units":"workspace units"},"action":{"keys":["delta_x","delta_y"],"units":"normalized [-1,1] Cartesian delta","action_scale":action_scale,"frequency_hz":100,"control_frame":"world_xy"},"expert":"greedy_scripted_v1"})
    ds.validate(); return ds

def save_dataset(dataset: Dataset, path: str | Path):
    dataset.validate(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={"metadata":np.asarray(json.dumps(dataset.metadata)),"count":np.asarray(len(dataset.episodes))}
    for i,e in enumerate(dataset.episodes):
        payload.update({f"ep{i}_obs":e.observations,f"ep{i}_actions":e.actions,f"ep{i}_rewards":e.rewards,f"ep{i}_success":np.asarray(e.success),f"ep{i}_timestamps":e.timestamps,f"ep{i}_metadata":np.asarray(json.dumps(e.metadata))})
    np.savez_compressed(path,**payload)
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); path.with_suffix(path.suffix+".sha256").write_text(digest+"\n")
    return digest

def load_dataset(path: str | Path, verify_checksum: bool=True) -> Dataset:
    path=Path(path)
    if verify_checksum:
        side=path.with_suffix(path.suffix+".sha256")
        if not side.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=side.read_text().strip(): raise ValueError("dataset checksum missing or does not match")
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"].item())); n=int(z["count"]); eps=[]
        for i in range(n):
            eps.append(Episode(z[f"ep{i}_obs"].copy(),z[f"ep{i}_actions"].copy(),z[f"ep{i}_rewards"].copy(),bool(z[f"ep{i}_success"].item()),z[f"ep{i}_timestamps"].copy(),json.loads(str(z[f"ep{i}_metadata"].item()))))
    ds=Dataset(eps,meta); ds.validate(); return ds

def split_episodes(dataset: Dataset, seed: int, train_fraction: float, validation_fraction: float):
    if train_fraction<=0 or validation_fraction<0 or train_fraction+validation_fraction>=1: raise ValueError("invalid split fractions")
    ids=np.random.default_rng(seed).permutation(len(dataset.episodes)); n=len(ids)
    ntrain=max(1,int(n*train_fraction)); nval=int(n*validation_fraction)
    if ntrain+nval>=n: nval=n-ntrain-1
    return ids[:ntrain].tolist(),ids[ntrain:ntrain+nval].tolist(),ids[ntrain+nval:].tolist()
