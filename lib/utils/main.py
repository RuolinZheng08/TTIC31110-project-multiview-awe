#!/usr/bin/env python3

import logging as log
import sys
log_filename = 'log_file.txt'
log.basicConfig(filename=log_filename, filemode='a', level=log.INFO, format="%(asctime)s: %(message)s")

# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logFormatter = log.Formatter("%(asctime)s: %(message)s")
consoleHandler = log.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.getLogger().addHandler(consoleHandler) # print to cmdline too

import os
import argparse
import random
import json
import numpy as np
import torch
import itertools as it

from utils.caller import call
from saver.saver import save_many, save_config, savez

def tuneHyperparamsAndSave(config, small_grid_search=True):
  if small_grid_search:
    # hyperparams = { # just for test
    #   'optim_lr' : [0.0001],
    #   'dropout' : [0.4],
    #   'loss_margin': [0.5]
    # }
    hyperparams = { # 4 combos, lr 0.0001 is too slow!!!
      'optim_lr' : [0.0001], # [0.001, 0.0001]
      'dropout' : [0.4],
      'loss_margin': [0.5] # [0.4, 0.5]
    }
    # tally:
    # obj02 phones: 0.001-0.4-0.4 and 0.001-0.4-0.5 have finished; 0.0001-0.4-0.5 in progress
    # obj02 chars:
    # obj0 phones: 0.001-0.4-0.4 started
    # obj0 chars: 0.001-0.4-0.4 started
  else:
    hyperparams = { # 40 combos
      'optim_lr' : [0.0001, 0.001],
      'dropout' : [0, 0.2, 0.4, 0.5],
      'loss_margin': [0.3, 0.4, 0.5, 0.6, 0.7]
    }

  params = ['optim_lr', 'dropout', 'loss_margin']
  candidates = list(it.product(*[hyperparams[param] for param in params]))

  for cand in candidates:
      (optim_lr, dropout, margin) = cand
      config.optim_lr = optim_lr
      config.net_view1_dropout = dropout
      config.net_view2_dropout = dropout
      config.loss_margin = margin
      config.global_global_step = 0

      log.info(f"Hyperparam tuning-------------------")
      log.info(f"optim_lr: {config.optim_lr}")
      log.info(f"dropout (for both): {config.net_view1_dropout}")
      log.info(f"margin: {config.loss_margin}")
      log.info(f"------------------------------------")

      if isinstance(config.global_step, int):
        random.seed(config.global_step)
        np.random.seed(config.global_step)
        torch.manual_seed(config.global_step)

      log.info(f"Calling main function: {config.main_fn}")
      log.info(f"Using config file: {config.config_file}")

      # print(config)
      # savez(f"save/dev-embs-{config.optim_lr}-{config.net_view1_dropout}-{config.loss_margin}")
      call(config.main_fn)(config)

rank = os.environ.get("RANK", 0)
world_size = os.environ.get("WORLD_SIZE", 1)

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

config_file = args.config_file

log.info('\n\nStart >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
log.info(f"Using {world_size} GPU(s)")
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

  tuneHyperparamsAndSave(config)
