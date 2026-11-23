from __future__ import annotations
import time
import numpy as np
from robot_benchmark.environments import ReachingEnv, ReachingSpec

def evaluate(policy, backend, seeds, max_steps, action_scale=.05, goal_radius=.06):
    env=ReachingEnv(backend,max_steps,ReachingSpec(action_scale=action_scale,goal_radius=goal_radius)); rows=[]
    for seed in seeds:
        obs,_=env.reset(seed=int(seed)); policy.reset(); lat=[]; success=False
        for step in range(max_steps):
            t0=time.perf_counter(); action=policy.act(obs); lat.append(time.perf_counter()-t0)
            obs,_,term,trunc,info=env.step(action)
            if term or trunc: success=info["success"]; break
        rows.append({"seed":int(seed),"success":bool(success),"steps":step+1,"final_distance":float(info["distance"]),"mean_inference_s":float(np.mean(lat)),"p95_inference_s":float(np.percentile(lat,95))})
    env.close()
    return {"episodes":rows,"success_rate":float(np.mean([r["success"] for r in rows])),"n":len(rows),"mean_steps":float(np.mean([r["steps"] for r in rows])),"mean_inference_s":float(np.mean([r["mean_inference_s"] for r in rows]))}
