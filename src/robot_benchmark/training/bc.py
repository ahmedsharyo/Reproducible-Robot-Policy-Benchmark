from __future__ import annotations
import copy, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from robot_benchmark.policies import BCModel

def _windows(episodes):
    return np.concatenate([e.observations[:-1] for e in episodes]), np.concatenate([e.actions for e in episodes])

def train_bc(train_episodes, val_episodes, config, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith("mps") and torch.backends.mps.is_available(): dev=torch.device("mps")
    elif device=="auto" and torch.backends.mps.is_available(): dev=torch.device("mps")
    elif device.startswith("cuda") and not torch.cuda.is_available(): dev=torch.device("cpu")
    else: dev=torch.device("cpu" if device=="auto" else device)
    tx,ty=_windows(train_episodes); vx,vy=_windows(val_episodes or train_episodes)
    om,os=tx.mean(0),tx.std(0); am,ast=ty.mean(0),ty.std(0); os=np.maximum(os,1e-6); ast=np.maximum(ast,1e-6)
    def norm(x,m,s): return torch.tensor((x-m)/s,dtype=torch.float32)
    loader=DataLoader(TensorDataset(norm(tx,om,os),norm(ty,am,ast)),batch_size=int(config["batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(seed))
    model=BCModel(4,2,tuple(config["hidden_sizes"])).to(dev); opt=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"]))
    best=float("inf"); best_state=None; patience=8; stale=0; history=[]; started=time.perf_counter()
    vxt=norm(vx,om,os).to(dev); vyt=norm(vy,am,ast).to(dev)
    for epoch in range(int(config["epochs"])):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val=float(torch.nn.functional.mse_loss(model(vxt),vyt).item())
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val})
        if val<best: best=val; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(best_state)
    return model,{"obs_mean":om,"obs_std":os,"action_mean":am,"action_std":ast},history,time.perf_counter()-started,str(dev),sum(p.numel() for p in model.parameters())
