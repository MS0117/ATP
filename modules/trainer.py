import random
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union, Tuple,Callable, Sized
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from torch.utils.data import Sampler
from trl import  GRPOTrainer
from trl.trainer.utils import disable_dropout_in_model
from trl.import_utils import is_rich_available, is_vllm_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from .reward_function import lean4_value_reward
from trl.core import masked_mean, masked_whiten
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

if is_deepspeed_available():
    import deepspeed

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
            self,
            data_source: Sized,
            mini_repeat_count: int,
            batch_size: int = 1,
            repeat_count: int = 1,
            seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



class CUSTOMTrainer(GRPOTrainer):
    def __init__(self, model, reward_funcs, args=None, **kwargs):

        if reward_funcs is not None and not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

            # 2. Loop over reward_funcs
        for i, reward_func in enumerate(reward_funcs):
            # If the reward_func is a string
            if isinstance(reward_func, str):
                # If that string is "lean", replace with your custom function
                if reward_func.lower() == "lean":
                    reward_funcs[i] = lean4_value_reward
                else:
                    # Otherwise, assume it's a valid HF model name
                    reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1
                    )
        #self.reward_funcs = reward_funcs in GRPO trainer
        super().__init__(model, reward_funcs, args=args, **kwargs)

        # If you want, store them directly for convenience:
        self.kl_coef = args.kl_coef
        self.loss_function = args.loss_function
        self.rloo_token_level=args.rloo_token_level
        self.kl_coef=args.kl_coef
        self.model_name=model
        self.normalize_advantage=args.normalize_advantage
        self.whiten_rewards=args.whiten_rewards
        self.cliprange=args.cliprange

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        #print("tokenizer_len",len(self.processing_class))
        prompts = [x["prompt"] for x in inputs]
        #print("promts",prompts)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        if 'gpt2' in str(self.model_name).lower() or 'llama' in str(self.model_name).lower() :
            self.processing_class.pad_token_id=self.processing_class.eos_token_id   #only for gpt2
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)       #pad_token=self.processing_class.eos_token_id
        prompt_inputs =  Trainer._prepare_inputs(self,prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                completions_text=[]
                #print("self.processing_class.eos_token_id",self.processing_class.eos_token_id)
                #print("self.processing_class.bos_token_id",self.processing_class.bos_token_id)
                for outputs in all_outputs:
                    for i,output in enumerate(outputs.outputs):
                        #print("index",i)
                        #print("vllm_len(output.token_ids",len(output.token_ids))
                        #print("vllm_output.token_ids",output.token_ids)
                        completion_ids.append(output.token_ids)             #no BOS token in output_token_ids
                        completions_text.append(output.text)
            else:
                #print("error")
                completion_ids = [None] * len(all_prompts_text)
                completions_text = [None] * len(all_prompts_text)

            #print("completion_ids:", completion_ids)
            #completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            print("completions_text",completions_text)
            print("total_compeltion_num",len(completions_text))
            completions_text= broadcast_object_list(completions_text, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            print("len(prompts)",len(prompts))
            print("self.accelerator.process_index",self.accelerator.process_index)
            print("process_slice",process_slice)
            #print("completions_text",completions_text)
            completions_text = completions_text[process_slice]
            print("sliced_completions_text", completions_text)

            #vllm_tokenizer = self.llm.get_tokenizer()
            encoded = self.processing_class(                #why? llm.out id is different from this encoded, I have to decode 그리고 넣어야함..decode token 자리마다 점수,, 이게 ccompletion_id from llm.generate와 달랐음
                completions_text,
                return_offsets_mapping=True,
                add_special_tokens=False)
            completion_ids = encoded["input_ids"]
            completion_ids = [tuple(ids) for ids in completion_ids]
            #for i in range(len(completion_ids)):
            #    print("index",i)
                #print("completion_ids",completion_ids[i])


            #check token id
            max_token_id = max(max(seq) for seq in completion_ids)  # or reencoded_ids_list
            #print("Max token id:", max_token_id)
            #print("Model vocab size:", self.model.config.vocab_size)
            assert max_token_id < self.model.config.vocab_size, "Token IDs out of range!"

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            """
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            """
            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            print("completion_ids",completion_ids.size())
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]



        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)

        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        """ until EOS
        completion_mask = [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ]
        """

        """ Until EOS+1
        value_mask = [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
        """

        values = [torch.zeros_like(ids) for ids in completion_ids]
        value_mask = (sequence_indices <= (eos_idx + 1).unsqueeze(1)).int()
        return_mask = value_mask
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        #print("prompt_completion_ids shape:", prompt_completion_ids.shape)
        #print("attention_mask shape:", attention_mask.shape)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                #print("old_per_token_logps_start")
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
                #print("old_per_token_logps",old_per_token_logps)
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        """
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        vllm_tokenizer=self.llm.get_tokenizer()
        encoded = vllm_tokenizer(
            completions_text,
            return_offsets_mapping=True,
            add_special_tokens=False)
        input_ids = encoded["input_ids"]
        for i in range(len(input_ids)):
            print("index",i)
            print("tokenized_input",input_ids[i])
            print("length",len(input_ids[i]))
        #completions_text=self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        """
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        #self.reward_funcs=reward_funcs=[lean4_value_reward] function...
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):

            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)



            elif "lean" in str(reward_func).lower():
                #print("prompts",prompts)
                #print("completions",completions)
                output_reward_func,binary_pass_score = reward_func(prompts=prompts, completions=completions,
                                                 processing_class=self.processing_class)  # reward feedback generation, lean4_scheduler

                """padded_scores tensor([[ 1.,  1.,  , -1., -1.,  1.],
                                        [1.,  1.,  , -1., -1.,  1.]])
                """
                binary_pass_score=torch.tensor(binary_pass_score, dtype=torch.float32, device=device)
                tactic_advantage = output_reward_func.to(dtype=torch.float32, device=device)  # reward
                #print("rewards_per_func.size()",rewards_per_func.size())


            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        """
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        """

        if self.whiten_rewards:
            tactic_advantage = masked_whiten(tactic_advantage, mask=value_mask, shift_mean=False)
            tactic_advantage = tactic_advantage * value_mask

        #print("rewards_per_func",tactic_advantage.size())
        print(" self.accelerator.num_processes", self.accelerator.num_processes)
        print(f"{self.accelerator.process_index}_prompt",prompts)
        print(f"{self.accelerator.process_index}_completions_text", completions_text)

        binary_pass_score = gather(binary_pass_score)

        # Apply weights to each reward function's output and sum






        # Compute grouped-wise rewards
        mean_grouped_rewards = binary_pass_score.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = binary_pass_score.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        binary_pass_score_advantage = binary_pass_score - mean_grouped_rewards
        if self.args.scale_rewards:
            binary_pass_score_advantage = binary_pass_score_advantage / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        binary_pass_score = binary_pass_score_advantage[process_slice]


        #print("rewards_per_func", rewards_per_func.size())



        #rewards_per_func = gather(rewards_per_func)
        #print("rewards_per_func_Gather",rewards_per_func.size())


        """
        for i in reversed(range(len(rewards_per_func.shape[-1]))):
            next_values = values[:, i + 1] if i < len(
                rewards_per_func.shape[-1]) - 1 else 0.0  # values=return in one trajectory environment
            values[:, i] = rewards_per_func[:, i] + args.gamma * next_values
        
        #values = values * value_mask

        # advantage function (Simple, GAE)

        advantages = values

        advantages = masked_whiten(advantages, completion_mask)
        advantages = advantages * completion_mask
        """




        tactic_advantage=tactic_advantage





        """
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        #print("process_slice",process_slice)
        advantages = advantages[process_slice]

        print("advantages", advantages.size())

        #print("advantages.size()", advantages.size())
        """






        torch.cuda.empty_cache()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"


        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["mean_completion_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["min_completion_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["max_completion_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["clipped_completions_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["mean_terminated_completion_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["min_terminated_completion_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["max_terminated_completion_length"].append(term_completion_mask.float().max().item())

        





        tactic_advantage_mean = tactic_advantage.mean()
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}_tactic_adv"].append(tactic_advantage_mean.item() )
            self._metrics[mode][f"rewards/{reward_func_name}_binary_mean"].append(mean_grouped_rewards.mean.item())
            self._metrics[mode][f"rewards/{reward_func_name}_binary_std"].append(std_grouped_rewards.mean().item())

        # rewards <-reward_per_func
        tactic_advantage_mean= tactic_advantage_mean
        #print("rewards",rewards)
        self._metrics[mode]["reward"].append(tactic_advantage_mean.item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "tactic_advantages": tactic_advantage,
            "binary_score": binary_pass_score
        }

    #@profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        num_batch = len(inputs["prompt_ids"])
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        ref_per_token_logps = inputs["ref_per_token_logps"]
        tactic_advantages = inputs["tactic_advantages"]
        binary_score=inputs["binary_score"]
        #print("completion_ids.size",completion_ids.size())
        #print("completion_mask.size",completion_mask.size())
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)




        if 'ppo' in self.loss_function.lower():
            ratio = torch.exp(per_token_logps - old_per_token_logps)
            pg_losses = -tactic_advantages * ratio
            pg_losses2 = -tactic_advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
            pg_loss_max = torch.max(pg_losses, pg_losses2)

            pg_loss = (pg_loss_max * completion_mask).sum() / completion_mask.sum()
            loss = pg_loss

        elif 'rloo' in self.loss_function.lower():  # token level
            # Token-level KL penalty: apply KL penalty per token
            kl = ref_per_token_logps - per_token_logps
            kl_reward = self.kl_coef * kl

            reward = advantages + kl_reward   #why?

            if self.rloo_token_level:
                #print("num_batch",num_batch)
                #print("self.num_generations",self.num_generations)
                reward = reward.view(num_batch // self.num_generations, self.num_generations, -1)
                baseline = (reward.sum(dim=1, keepdim=True) - reward) / (self.num_generations - 1.0)
                # shape is still [B, num_generations, T]

                advantages = reward - baseline
                advantages = advantages.view(num_batch, -1)

            else:  # Vanilla RLOO
                reward = reward.view(num_batch, self.num_generations, -1)
                seq_reward = reward.sum(dim=2)  # shape [B, num_generations]
                baseline = (seq_reward.sum(dim=1, keepdim=True) - seq_reward)(num_generations - 1.0)
                advantages = reward - baseline
                advantages = advantages.view(num_batch, -1)

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
            pg_loss_max = torch.max(pg_losses, pg_losses2)

            pg_loss = (pg_loss_max * completion_mask).sum() / completion_mask.sum()
            loss = pg_loss



        elif 'reinforce' in self.loss_function.lower():  # baseline?

            advantages=tactic_advantages
            kl = ref_per_token_logps - per_token_logps
            kl_reward = self.kl_coef * kl

            advantages = advantages + kl_reward

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
            pg_loss_max = torch.max(pg_losses, pg_losses2)

            pg_loss = (pg_loss_max * completion_mask).sum() / completion_mask.sum()
            loss = pg_loss



        elif 'grpo' in self.loss_function.lower():  # baseline?
            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = (
                        torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

            # Compute the loss
            tactic_advantages = inputs["tactic_advantages"]
            binary_score = inputs["binary_score"]
            advantages = tactic_advantages+binary_score.unsqueeze(1)
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
            # _generate_and_score_completions) and use per_token_logps.detach() instead.
            old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()



        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (pg_losses < pg_losses2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        #print("self._metrics[mode]",self._metrics[mode])
        #print("loss",loss)
        return loss
