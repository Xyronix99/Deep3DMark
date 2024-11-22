import os
import argparse
import mlconfig
import shutil
import logging
import torch
import torch.distributed as dist

from tasks import *
from model import *
from util import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    '''cfg setup'''
    args = parse_args()
    cfg = mlconfig.load(args.config)

    V().name = os.path.basename(args.config)
    V().cfg = cfg
    os.makedirs(os.path.join(cfg.output_dir, V().name), exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.output_dir, V().name))

    '''logger'''
    V().logger = logging.getLogger("")
    V().logger.addHandler(logging.FileHandler(os.path.join(cfg.output_dir, V().name, 'my.log')))
    V().logger.setLevel(logging.INFO)

    '''torch setup'''
    gpu_id = 5#reserve_gpu(None)
    device_id = 0
    V().who_am_I = os.getpid()
    V().device = device_id
    trainer = cfg.trainer()

    torch.cuda.set_device(device_id)
    V().info(f'Selecting device {device_id}')

    # mesh = pyvista.PolyData("demo/generalization/bunny.ply")
    # mesh.plot()
    '''debug'''
    trainer.debug()