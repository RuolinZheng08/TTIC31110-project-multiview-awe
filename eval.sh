#!/bin/bash

echo HOSTNAME: $(hostname)

export PYTHONPATH=src:lib

lib/utils/main.py eval_config.json
