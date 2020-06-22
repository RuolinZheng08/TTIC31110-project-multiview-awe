#!/usr/bin/env python3.7

import logging as log
log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")

import os
import argparse
import random
import json
import numpy as np
import torch

from utils.caller import call

rank = os.environ.get("RANK", 0)
world_size = os.environ.get("WORLD_SIZE", 1)

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

config_file = args.config_file

if torch.cuda.is_available():
  log.info(f"Using {world_size} GPU(s)")
else:
  log.info("Using CPU")
log.info(f"Machine: {rank} / {world_size}")

if world_size > 1:
  log.info(f"Master port: {os.environ['MASTER_PORT']}")
  log.info(f"Master address: {os.environ['MASTER_ADDR']}")
  torch.distributed.init_process_group("nccl")
  torch.distributed.barrier()

with open(config_file, "r") as f:
  config = argparse.Namespace(**json.load(f))
  if not hasattr(config, "config_file"):
    config.config_file = config_file

if isinstance(config.global_step, int):
  random.seed(config.global_step)
  np.random.seed(config.global_step)
  torch.manual_seed(config.global_step)

log.info(f"Calling main function: {config.main_fn}")
log.info(f"Using config file: {config.config_file}")

call(config.main_fn)(config)