#!/bin/bash
#SBATCH --signal=B:SIGUSR1@300


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=GRPO accelerate launch --config_file ./accelerate/deepspeed3_2.yaml \
    train.py ./configs/grpo.yaml