import argparse
from robot_benchmark.cli import run
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--device",default="auto"); a=p.parse_args(); run(a.config,a.device)
