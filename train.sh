#!/bin/bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=CUSTOM accelerate launch --config_file ./accelerate/deepspeed3.yaml \
    train.py ./configs/symr.yaml