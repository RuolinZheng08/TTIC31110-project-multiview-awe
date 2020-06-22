#!/bin/bash

echo HOSTNAME: $(hostname)

export PYTHONPATH=src:lib

lib/utils/main.py train_config.json