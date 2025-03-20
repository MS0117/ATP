#!/bin/bash
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=CUSTOM accelerate launch --config_file ./accelerate/multi.yaml \
    train.py ./configs/symr_peft_test.yaml