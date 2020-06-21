#!/bin/bash

echo HOSTNAME: $(hostname)

export PYTHONPATH=src:lib

python3.7 lib/utils/main.py eval_config.json
