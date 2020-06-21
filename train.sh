#!/bin/bash

echo HOSTNAME: $(hostname)

export PYTHONPATH=src:lib

python3.7 lib/utils/main.py train_config.json