from __future__ import annotations
import argparse, json, platform, random, time
from pathlib import Path
import numpy as np
import torch, yaml
from robot_benchmark import __version__
from robot_benchmark.datasets import generate_dataset, save_dataset, split_episodes
from robot_benchmark.evaluation import evaluate
from robot_benchmark.policies import BCPolicy, BCModel
from robot_benchmark.training import train_bc

def run(config_path, device="auto"):
    cfg=yaml.safe_load(Path(config_path).read_text()); run_id=cfg.get("run_id",time.strftime("%Y%m%d-%H%M%S")); out=Path(cfg.get("output_dir","results"))/run_id; out.mkdir(parents=True,exist_ok=True)
    env=cfg["environment"]; dcfg=cfg["data"]
    ds=generate_dataset(int(dcfg["episodes"]),int(dcfg["seed"]),env["backend"],int(env["max_steps"]),float(env["action_scale"]),float(env["goal_radius"]))
    ds_path=out/"demonstrations.npz"; checksum=save_dataset(ds,ds_path)
    tr,va,te=split_episodes(ds,int(dcfg["seed"]),float(dcfg["train_fraction"]),float(dcfg["validation_fraction"]))
    (out/"splits.json").write_text(json.dumps({"train":tr,"validation":va,"test":te},indent=2))
    versions={"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__}
    try:
        import mujoco; versions["mujoco"]=mujoco.__version__
    except ImportError: versions["mujoco"]=None
    results=[]
    for seed in cfg["training"]["training_seeds"]:
        model,norm,history,train_s,actual_device,nparams=train_bc([ds.episodes[i] for i in tr],[ds.episodes[i] for i in va],cfg["training"],int(seed),device)
        policy=BCPolicy(model,norm["obs_mean"],norm["obs_std"],norm["action_mean"],norm["action_std"],actual_device)
        ev=cfg["evaluation"]; eval_seeds=[int(s) for s in ev["seeds"][:int(ev["episodes"])]]
        if not eval_seeds:
            eval_seeds=list(range(50000,50000+int(ev["episodes"])))
        metrics=evaluate(policy,env["backend"],eval_seeds,int(ev["max_steps"]),float(env["action_scale"]),float(env["goal_radius"]))
        ckpt=out/f"bc_seed_{seed}.pt"
        torch.save({"model":model.state_dict(),"normalization":{k:torch.as_tensor(v) for k,v in norm.items()},"config":cfg,"seed":int(seed)},ckpt)
        results.append({"training_seed":int(seed),"train_wall_s":train_s,"device":actual_device,"parameter_count":nparams,"checkpoint_bytes":ckpt.stat().st_size,"history":history,"evaluation":metrics})
    summary={"schema_version":1,"run_id":run_id,"dataset_sha256":checksum,"dataset_episodes":len(ds.episodes),"train_ids":tr,"validation_ids":va,"held_out_ids":te,"versions":versions,"hardware":{"platform":platform.platform(),"mps_available":torch.backends.mps.is_available(),"device_requested":device},"config":cfg,"policy":"single_action_behavior_cloning","results":results}
    (out/"results.json").write_text(json.dumps(summary,indent=2));
    print(json.dumps({"output":str(out),"success_rates":[r["evaluation"]["success_rate"] for r in results],"dataset_sha256":checksum},indent=2)); return summary

def main():
    ap=argparse.ArgumentParser(prog="robot-benchmark"); sub=ap.add_subparsers(dest="command",required=True); p=sub.add_parser("run"); p.add_argument("--config",required=True); p.add_argument("--device",default="auto",choices=["auto","cpu","mps"]); args=ap.parse_args()
    if args.command=="run": run(args.config,args.device)
if __name__=="__main__": main()
