#!/bin/bash

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=SFT accelerate launch --config_file accelerate/deepspeed2.yaml \
    train.py /userhomes/minsu/symr/configs/sft.yaml