import argparse
from robot_benchmark.datasets import generate_dataset, save_dataset
p=argparse.ArgumentParser(); p.add_argument("--episodes",type=int,default=100); p.add_argument("--seed",type=int,default=0); p.add_argument("--output",default="demonstrations.npz"); p.add_argument("--backend",choices=["numpy","mujoco"],default="numpy"); a=p.parse_args()
ds=generate_dataset(a.episodes,a.seed,a.backend); print(save_dataset(ds,a.output))
