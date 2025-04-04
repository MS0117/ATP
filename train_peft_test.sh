#!/bin/bash

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=GRPO accelerate launch --config_file ./accelerate/multi.yaml \
    train.py ./configs/grpo.yaml