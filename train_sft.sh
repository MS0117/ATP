#!/bin/bash

CUDA_VISIBLE_DEVICE=0

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=SFT accelerate launch --config_file ./accelerate/sft.yaml \
    train.py ./configs/sft.yaml