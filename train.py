import os
import sys
import time
import torch
import torch.distributed as dist
import logging
import datasets
import wandb
import signal
import transformers
from typing import Dict, List
from accelerate import PartialState,Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import (

    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,

)

from modules import CUSTOMTrainer,CustomRLOOTrainer,lean4_value_reward,lean4_grpo_reward,lean4_rloo_reward,lean4_rloo_custom_reward
from utils import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    CUSTOMConfig,
    RLOOConfig,

    make_padded_logits,
    DTYPE_MAP,
    get_quantization_config,
    get_peft_config
)
from peft import get_peft_model
from trl import GRPOConfig,GRPOTrainer,RewardConfig,SFTConfig, SFTTrainer,PPOTrainer,PPOConfig, RewardTrainer, RLOOTrainer, RLOOConfig, DataCollatorForCompletionOnlyLM
#from src import ()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 16

tqdm.pandas()

logger = logging.getLogger(__name__)
START_TIME = time.strftime("%Y%m%d_%H%M%S")


"""
def handle_exit(signum, frame):
    logger.info("SIGTERM received. Finishing WandB and exiting safely.")
    wandb.finish()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_exit)
"""


LEAN4_DEFAULT_HEADER = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)


def build_sft_dataset(example):
    header = example.get("header", LEAN4_DEFAULT_HEADER)
    informal_prefix = example.get("informal_prefix", "")
    formal_statement = example.get("formal_statement", "")
    formal_proof = example.get("formal_proof", "")

    # (1) Prompt: 코드 블록 열기만!
    #    (닫는 백틱은 넣지 않는다)
    prompt_text = (
        "Complete the following Lean 4 code...\n\n"
        "```lean4\n"
        f"{header}"
        f"{informal_prefix}"
        f"{formal_statement}"
    )

    # (2) Completion: 여기서 코드 블록 닫기
    completion_text = (
        f"{formal_proof}\n"  # formal_proof 끝난 뒤 개행
        "```"  # 코드 블록 마무리
    )

    return {
        "prompt": prompt_text,
        "completion": completion_text
    }


def main(model_args,
         data_args,
         training_args,
         model_type: AutoModel,
         training_type: str
         ) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set run name for WandB
    training_args.run_name = f"{training_args.run_name}-{START_TIME}"


    #if'rloo' in training_type.lower() or 'rloocustom' in training_type.lower():
    #    from accelerate import Accelerator
    #    accelerator = Accelerator()

        # Load tokenizer and model
    print("trust",model_args.trust_remote_code )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left",trust_remote_code=model_args.trust_remote_code )
    #quantization_config=get_quantization_config(model_args)
    quantization_config = get_quantization_config(model_args)

    peft_config = get_peft_config(model_args)

    if training_type.lower() in ['sft', 'rloo', 'customrloo'] :
        model = model_type.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=DTYPE_MAP[model_args.torch_dtype],
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=quantization_config
        )
        if training_type.lower() in ['rloo', 'customrloo'] :
            if model_args.use_peft:
                policy = get_peft_model(model, peft_config)
            else:
                policy=model


            if training_args.gradient_checkpointing:
                policy.enable_input_require_grads()
            ref=model_type.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=DTYPE_MAP[model_args.torch_dtype],
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=quantization_config
        )

    if 'grpo' == training_type.lower() or 'custom'== training_type.lower():
        model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=DTYPE_MAP[model_args.torch_dtype],
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        training_args.model_init_kwargs = model_kwargs
        model=model_args.model_name_or_path
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if 'grpo' in training_type.lower() or 'custom' in training_type.lower() or 'rloo' in training_type.lower() or 'rloocustom' in training_type.lower():

        if 'minif2f' in data_args.dataset_name.lower():
            data_path="/userhomes/minsu/symr/data/toy_train.jsonl"

        if 'workbook' in data_args.dataset_name.lower():
            data_path="/userhomes/minsu/symr/data/train_dataset/training_dataset_random_15429.jsonl"

        if 'deepseek' in data_args.dataset_name.lower():
            data_path="/userhomes/minsu/symr/data/train_dataset/deepseek_rl_random_15429.jsonl"

    if 'sft' in training_type.lower():
        data_path = "/userhomes/minsu/symr/data/deepseek_sft.jsonl"

    # Load dataset
    data_files = {"train": data_path}
    raw_datasets = load_dataset("json", data_files=data_files)

    # If you also have a validation/test JSONL file, you can include:
    # data_files = {"train": "minif2f_train.jsonl", "test": "minif2f_test.jsonl"}
    # raw_datasets = load_dataset("json", data_files=data_files)

    # 2. Define a function to build a single "prompt" string



    def build_prompt(example):
        """Combine header + informal_prefix + formal_statement into a single prompt."""
        header = example.get("header", LEAN4_DEFAULT_HEADER)
        informal_prefix = example.get("informal_prefix", str())
        formal_statement = example.get("formal_statement", "")

        prompt_text = (
            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n"
            f"{header}"
            f"{informal_prefix}"
            f"{formal_statement}"
        )
        return {"prompt": prompt_text}

    # 3. Apply 'build_prompt' to the dataset
    #    This creates a "prompt" column that GRPOTrainer will use internally.
    # Map and filter dataset
    with PartialState().local_main_process_first():

        if 'sft' in training_type.lower():
            if training_type.lower() == "sft":
                train_dataset = raw_datasets["train"].map(
                    build_sft_dataset
                    # remove_columns=raw_datasets["train"].column_names  # 필요에 따라 제거
                )

        else:
            train_dataset = raw_datasets["train"].map(
            build_prompt,
            # Keep other columns if you need them; or remove them if not.
            # remove_columns=raw_datasets["train"].column_names
        )

    def tokenize_fn(batch):
        return tokenizer(
            batch["prompt"],
            max_length=512,
            truncation=True,
            add_special_tokens=True,
        )

    # 👉 RLOO 처리: prompt + tokenize
    if "rloo" in training_type.lower():
        with PartialState().local_main_process_first():
            train_dataset = raw_datasets["train"].map(build_prompt)

            train_dataset = train_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=[
                    "header", "informal_prefix", "formal_statement", "prompt"
                ],
            )



    eval_dataset = None if not data_args.use_test_set else test_dataset
    #reward_funcs=training_args.reward_type




    # Prepare trainer and train
    if  'sft' in training_type.lower():
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config
        )
    elif training_type == 'ppo':
        trainer = PPOTrainer(
            model=model,
            reward_model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config
        )
    elif  'custom'==training_type.lower():
        trainer = CUSTOMTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=training_args.reward_type,
            peft_config=peft_config
        )

    elif 'grpo' ==training_type.lower():

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=lean4_grpo_reward,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config
        )
    elif 'rloo' ==training_type.lower():

        trainer = RLOOTrainer(
            policy=policy,
            ref_policy=ref,
            processing_class=tokenizer,
            reward_model=lean4_rloo_reward,
            config=training_args,
            train_dataset=train_dataset,
        )
    elif 'customrloo'== training_type.lower():

        trainer = CustomRLOOTrainer(
            policy=policy,
            ref_policy=ref,
            processing_class=tokenizer,
            reward_model=lean4_rloo_custom_reward,
            config=training_args,
            train_dataset=train_dataset,

        )



    else:
        trainer = CriticTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )


    #slurm signal

    def graceful_shutdown(signum, frame):
        logging.warning(f"[Rank {trainer.accelerator.process_index}] Got signal {signum} → checkpoint & exit")
        print("handler_ shutdown")
        if trainer.accelerator.is_main_process:
            # ----- 2‑a  Save everything you need  -----------------------
            #  (choose ONE of the two styles below)

            ## Style A — ZeRO‑aware Accelerate (recommended)
            # accelerator.save_state("signal_ckpt")   # saves model + opt shards

            ## Style B — HF/TRL style
            #trainer.save_state("signal_ckpt")  # opt/sched/random state
            trainer.save_model(training_args.output_dir)  # weights + tokenizer

            wandb.finish()

        try:
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.free_memory()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")
        # ----- 2‑d  Exit the whole SLURM cgroup
        os.killpg(0, signal.SIGTERM)  # let children exit too
        os._exit(0)  # immediate, bypasses atexit

    def register_handlers():
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, graceful_shutdown)

    register_handlers()

    if training_args.resume_from_checkpoint:
        print("rusume")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("*** Training complete ***")

    # Save model
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if 'sft' in training_type.lower():
        kwargs = {
            # "finetuned_from": model_args.model_name_or_path,  # This argument is not supported
            "model_name": model_args.model_name_or_path,

        }
        # Save everything else on main process
    else:
        kwargs = {
            "model_name": model_args.model_name_or_path,
        }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Push to hub
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


    logger.info("*** Training complete! ***")
    #wandb.finish()
    #logger.info("WandB run finished cleanly.")


if __name__ == "__main__":
    # Select proper Config and Trainer class
    training_type = os.getenv("TRAINING_TYPE")
    if training_type == "SFT":
        config_type = SFTConfig
        model_type = AutoModelForCausalLM
    elif training_type == 'GRPO':
        config_type = GRPOConfig
        model_type = AutoModelForCausalLM

    elif training_type == 'CUSTOM':
        config_type = CUSTOMConfig
        model_type = AutoModelForCausalLM
    elif training_type == 'PPO':
        config_type = PPOConfig
        model_type = AutoModelForCausalLM
    elif training_type == 'RLOO':
        config_type = RLOOConfig
        model_type = AutoModelForCausalLM
    elif training_type == 'CUSTOMRLOO':
        config_type = RLOOConfig
        model_type = AutoModelForCausalLM
    #elif training_type == 'RLOO_CUSTOM':
    #    config_type = RLOOCUSTOMConfig
    #    model_type = AutoModelForCausalLM


    else:
        raise Exception("Please check the training method.")
    parser = H4ArgumentParser((ModelArguments, DataArguments, config_type))
    model_args, data_args, training_args = parser.parse()

    if model_args.torch_dtype == 'bf16':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set up WandB is needed
    if data_args.wandb_entity is not None and data_args.wandb_project is not None:
        os.environ["WANDB_ENTITY"] = data_args.wandb_entity
        os.environ["WANDB_PROJECT"] = data_args.wandb_project

    # Start training
    main(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model_type=model_type,
        training_type=training_type
    )