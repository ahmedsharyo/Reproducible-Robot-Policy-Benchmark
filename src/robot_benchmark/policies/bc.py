from __future__ import annotations
import torch
from torch import nn

class BCModel(nn.Module):
    def __init__(self, observation_dim=4, action_dim=2, hidden_sizes=(128,128)):
        super().__init__(); layers=[]; d=observation_dim
        for h in hidden_sizes: layers.extend([nn.Linear(d,h),nn.ReLU()]); d=h
        layers.append(nn.Linear(d,action_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class BCPolicy:
    def __init__(self, model, obs_mean, obs_std, action_mean, action_std, device="cpu"):
        self.model=model.to(device).eval(); self.device=device
        self.obs_mean=torch.as_tensor(obs_mean,dtype=torch.float32,device=device); self.obs_std=torch.as_tensor(obs_std,dtype=torch.float32,device=device).clamp_min(1e-6)
        self.action_mean=torch.as_tensor(action_mean,dtype=torch.float32,device=device); self.action_std=torch.as_tensor(action_std,dtype=torch.float32,device=device).clamp_min(1e-6)
    def reset(self): pass
    @torch.inference_mode()
    def predict(self, observation):
        x=torch.as_tensor(observation,dtype=torch.float32,device=self.device)
        y=self.model((x-self.obs_mean)/self.obs_std)*self.action_std+self.action_mean
        return y.clamp(-1,1).cpu().numpy()
    def act(self, observation): return self.predict(observation)
