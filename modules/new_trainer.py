import random
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union, Tuple, Callable, Sized
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
import csv
import os
import math
from torch.utils.data import Sampler
from trl import GRPOTrainer
from trl.trainer.utils import disable_dropout_in_model
from trl.import_utils import is_rich_available, is_vllm_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from .reward_function import lean4_value_reward,lean4_rloo_custom_reward,deepseek_lean4_rloo_custom_reward
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


class NEWCUSTOMTrainer(GRPOTrainer):
    def __init__(self, model, reward_funcs, args=None, **kwargs):

        if reward_funcs is not None and not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

            # 2. Loop over reward_funcs
        for i, reward_func in enumerate(reward_funcs):
            # If the reward_func is a string
            if isinstance(reward_func, str):
                # If that string is "lean", replace with your custom function
                if reward_func.lower() == "lean":
                    if 'deepseek' in model.lower():
                        reward_function =deepseek_lean4_rloo_custom_reward
                        reward_funcs[i] = reward_function
                    else:
                        reward_function =lean4_rloo_custom_reward
                        reward_funcs[i] = reward_function


                else:
                    # Otherwise, assume it's a valid HF model name
                    reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1
                    )
        # self.reward_funcs = reward_funcs in GRPO trainer
        super().__init__(model, reward_funcs, args=args, **kwargs)

        # If you want, store them directly for convenience:
        self.kl_coef = args.kl_coef
        self.loss_function = args.loss_function
        self.rloo_token_level = args.rloo_token_level
        self.kl_coef = args.kl_coef
        self.model_name = model
        self.normalize_advantage = args.normalize_advantage
        self.whiten_rewards = args.whiten_rewards
        self.cliprange = args.cliprange
        self.negative_dropout = args.negative_dropout
        self.dropout_rate = args.dropout_rate
        self.alpha_advantage = args.alpha_advantage
        self.delta0=args.delta0
        self.delta1= args.delta1
        self.delta2= args.delta2
        self.adv_baseline= args.adv_baseline
        self.score_assign= args.score_assign
        self.adv_method= args.adv_method
        self.parse_method= args.parse_method
        self.weighted_adv= args.weighted_adv
        self.first_error=args.first_error
        self.delta_clip=args.delta_clip
        self.potent_func= args.potent_func
        self.potent_type= args.potent_type
        self.potent_positive= args.potent_positive
        self.shift_potential= args.shift_potential
        self.potent_coef= args.potent_coef
        self.entropy_adv=args.entropy_adv
        self.entropy_reg=args.entropy_reg
        self.entropy_coef=args.entropy_coef
        self.entropy_position=args.entropy_position
        self.asymmetric= args.asymmetric
        self.tactic_distribution=args.tactic_distribution
        self.distribution_method=args.distribution_method
        self.positive_entropy_drop=args.positive_entropy_drop
        self.divide_weight_method=args.divide_weight_method
        self.first_tactic_token=args.first_tactic_token
        self.advantage_distribute_top_k=args.advantage_distribute_top_k
        self.reward_uniform_distribution=args.reward_uniform_distribution
        self.only_first_error_distribute=args.only_first_error_distribute
        self.weighted_prob_advantage=args.weighted_prob_advantage

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

    @profiling_decorator
    def entropy_from_logits(self,logits, chunk_size: int = 1) -> torch.Tensor:
        """
        Compute the Shannon entropy (in nats) for each row of *logits* without
        materialising the full soft-max in memory.
        The batch dimension is processed in chunks of size `chunk_size` so that
        only a subset of rows is expanded to probabilities at any one time.
        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
                leading dimensions are preserved.
            chunk_size (`int`, *optional*, defaults to `1`):
                Number of rows to process per iteration.
        Returns:
            `torch.Tensor`:
                Entropy values with shape `logits.shape[:-1]`.
        """
        per_token_entropies = []
        for logits_chunk in logits.split(chunk_size, dim=0):
            logps = F.log_softmax(logits_chunk, dim=-1)
            chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
            per_token_entropies.extend(chunk_entropy)

        per_token_entropies = torch.stack(per_token_entropies)
        return per_token_entropies

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        entropies = self.entropy_from_logits(logits)
        #print("entropy",entropies[0])
        return selective_log_softmax(logits, input_ids), entropies  # compute logprobs for the input tokens


    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # print("tokenizer_len",len(self.processing_class))
        prompts = [x["prompt"] for x in inputs]
        # print("promts",prompts)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        if 'gpt2' in str(self.model_name).lower() or 'llama' in str(self.model_name).lower():
            self.processing_class.pad_token_id = self.processing_class.eos_token_id  # only for gpt2

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left",
            add_special_tokens=False)  # pad_token=self.processing_class.eos_token_id
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # print("max_prompt_length",self.max_prompt_length)
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
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
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

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                # print("old_per_token_logps_start")
                old_per_token_logps,entropies = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
                # print("old_per_token_logps",old_per_token_logps)
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps,entropies = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps,entropies = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        #print("completion_ids",completion_ids[0])
        #completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces = False)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # self.reward_funcs=reward_funcs=[lean4_value_reward] function...
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
                # print("prompts",prompts)
                # print("completions",completions)
                output_reward_func, binary_pass_score,all_token_ids,tactic_mean,adv_masks,all_tactic_ids,all_first_error_masks,timeout_flags = reward_func(prompts=prompts, completions=completions,
                                                                    processing_class=self.processing_class,
                                                                    max_len=completion_ids.size(
                                                                        1), num_generation=self.num_generations,
                delta0=self.delta0,
                delta1=self.delta1,
                delta2=self.delta2 ,
                adv_baseline=self.adv_baseline,
                score_assign=self.score_assign ,
                adv_method=self.adv_method ,
                parse_method=self.parse_method,
                first_error=self.first_error,
                potent_func=self.potent_func,
                potent_type=self.potent_type,
                potent_positive  =self.potent_positive,
                shift_potential=self.shift_potential,
                potent_coef=self.potent_coef,
                entropy_position=self.entropy_position
                )  # reward feedback generation, lean4_scheduler

                """padded_scores tensor([[ 1.,  1.,  , -1., -1.,  1.],
                                        [1.,  1.,  , -1., -1.,  1.]])
                """
                binary_pass_score = torch.tensor(binary_pass_score, dtype=torch.float32, device=device)
                tactic_advantage = torch.tensor(output_reward_func, dtype=torch.float32, device=device)  # reward
                adv_masks = torch.tensor(adv_masks, dtype=torch.float32, device=device)
                all_tactic_ids=torch.tensor(all_tactic_ids, dtype=torch.float32, device=device)
                all_first_error_masks=torch.tensor(all_first_error_masks, dtype=torch.float32, device=device)
                timeout_flags = torch.tensor(timeout_flags, dtype=torch.float32, device=device)
                #print("rewards_per_func.size()",tactic_advantage.size())


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

        """
        if self.whiten_rewards:
            tactic_advantage = masked_whiten(tactic_advantage, mask=completion_mask, shift_mean=False)
            tactic_advantage = tactic_advantage * completion_mask

        """

        # normalized seperately

        #tactic_advantage = masked_whiten(tactic_advantage, mask=completion_mask, shift_mean=False)
        tactic_advantage = tactic_advantage * completion_mask
        masked_entropy = entropies * completion_mask
        #print("masked_entropy",masked_entropy[0])

        # print("rewards_per_func",tactic_advantage.size())
        # print(" self.accelerator.num_processes", self.accelerator.num_processes)
        # print(f"{self.accelerator.process_index}_prompt",prompts)
        # print(f"{self.accelerator.process_index}_completions_text", completions_text)

        binary_reward=binary_pass_score

        binary_pass_score = gather(binary_pass_score)
        tactic_advantage_mean=tactic_advantage.mean()
        gathered_tactic_advantage_mean=gather(tactic_advantage_mean).mean()
        gatherd_tactic_mean=gather(torch.tensor(tactic_mean,dtype=torch.float32, device=device)).mean()
        #gathered_tactic_mean=gather(tactic_mean)




        #entropy
        token_counts = completion_mask.sum(-1).clamp(min=1)
        mean_entropy_per_seq = masked_entropy.sum(-1) / token_counts  # (B,)

        # Batch mean entropy (scalar)
        gathered_mean_entropy_per_seq = gather(mean_entropy_per_seq)
        gathered_mean_entropy = gathered_mean_entropy_per_seq.mean()


        #torch.set_printoptions(threshold=float('inf'))

        #print("rewards ", binary_pass_score )





        binary_score_mean = binary_pass_score.mean(0)
        # Apply weights to each reward function's output and sum

        # Compute grouped-wise rewards
        mean_grouped_rewards = binary_pass_score.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = binary_pass_score.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        binary_pass_score_advantage = binary_pass_score - mean_grouped_rewards

        binary_pass_score_advantage = binary_pass_score_advantage / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        binary_pass_score = binary_pass_score_advantage[process_slice]

        # print("rewards_per_func", rewards_per_func.size())

        # rewards_per_func = gather(rewards_per_func)
        # print("rewards_per_func_Gather",rewards_per_func.size())

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
        self._metrics[mode][f"entropy"].append(gathered_mean_entropy.item())
        reward_mean = binary_score_mean

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_mean.item())
        self._metrics[mode][f"rewards/tactic_mean"].append(gatherd_tactic_mean.item())
        self._metrics[mode][f"rewards/tactic_adv"].append(gathered_tactic_advantage_mean.item())
        self._metrics[mode][f"rewards/grouped_binary_mean"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode][f"rewards/grouped_binary_std"].append(std_grouped_rewards.mean().item())

        # rewards <-reward_per_func
        tactic_advantage_mean = tactic_advantage_mean
        # print("rewards",rewards)
        self._metrics[mode]["reward"].append(tactic_advantage_mean.item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        agg_timeout_flags = self.accelerator.gather_for_metrics(timeout_flags)  # (N_total,)
        timeout_ratio = agg_timeout_flags.float().mean().item()

        # 로그
        self._metrics[mode]["timeout_ratio"].append(timeout_ratio)

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

        """
        print(
            f"prompt_ids       = {prompt_ids}\n"
            f"advantages       = {binary_pass_score}\n"
            f"completion_ids   = {completion_ids}\n"
            f"tactic_advantages= {tactic_advantage}",
            flush=True
        )
        """


        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "tactic_advantages": tactic_advantage,
            "binary_score": binary_pass_score,
            "adv_mask":adv_masks,
            "all_tactic_ids":all_tactic_ids,
            "all_first_error_masks":all_first_error_masks,
            "binary_reward":binary_reward
        }




    def distribute_tactic_reward_with_ids(self,
            per_token_logps: torch.Tensor,  # (B, L)
            old_per_token_logps: torch.Tensor,  # (B, L)
            completion_mask: torch.Tensor,  # (B, L)  bool / 0‑1
            tactic_ids: torch.Tensor,  # (B, L)  ‑1 = no tactic, 0..M_j‑1 otherwise
            tactic_token_adv: torch.Tensor,  # (B, L)  SAME SHAPE, each tactic’s tokens carry same value
            entropies: torch.Tensor,
            all_first_error_masks,
            eps: float = 1e-8,

    ):
        """
        Returns
        -------
        token_advantages : (B, L)
            Re‑distributed advantages whose *sum* over a tactic segment equals the
            segment’s scalar advantage, but apportioned according to |1‑π/π_old|.
        weights : (B, L)
            The normalised I_t inside each tactic (0 outside).
        """
        tactic_ids = tactic_ids.long()


        #entropy

        if "info_gain" in self.distribution_method.lower():

            H_prev = torch.zeros_like(entropies)  # (B, L)
            H_prev[:, 1:] = entropies[:, :-1]

            I_signed = H_prev - entropies  # ΔH = H_{t-1} - H_t

            I_positive = torch.clamp(I_signed, min=0.0)
            I_abs = I_signed.abs()

            I_t = I_positive if self.positive_entropy_drop else I_abs

        # === 2) entropy 레벨 자체 ==========================
        elif "entropy" in self.distribution_method.lower():  # 혹은 `in self.entropy_level` 등
            I_t = entropies.clone()

        I_t = I_t.float()
        """ token ratio between policies
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # (B, L)
        I_t = torch.abs(1.0 - ratio) * completion_mask.float()  # (B, L)
        """



        #print("I_t[0]:", I_t[0].tolist())
        B, L = I_t.shape
        token_adv_out = torch.zeros_like(I_t)
        weights_out = torch.zeros_like(I_t)

        for b in range(B):
            ids_b = tactic_ids[b]  # (L,)
            #print("ids_b",ids_b)
            It_b = I_t[b]
            adv_b = tactic_token_adv[b]

            tactic_tokens = ids_b >= 0
            if not tactic_tokens.any():
                continue

            ids_valid = ids_b[tactic_tokens]  # (n,)
            #print("ids_valid", ids_valid)
            It_valid = It_b[tactic_tokens]
            adv_valid = adv_b[tactic_tokens]

            # How many distinct tactic ids in this sequence?
            M_j = int(ids_valid.max().item()) + 1

            # ------------------------------------------------------------------
            # 1) tactic scalar advantage = DISTINCT VALUE carried by its tokens.
            #    Here we simply take the first occurrence (they’re identical).
            # ------------------------------------------------------------------
            # Each tactic id will store its scalar advantage here:
            """
            r_tactic = torch.zeros(M_j, device=It_b.device, dtype=It_b.dtype)
            # Because tokens of the same tactic carry the same value,
            # we can use index_add_ and then divide by counts to pick it.
            r_sum = torch.zeros_like(r_tactic)
            counts = torch.zeros_like(r_tactic)
            r_sum.index_add_(0, ids_valid, adv_valid)
            counts.index_add_(0, ids_valid, torch.ones_like(adv_valid))
            r_tactic = r_sum / counts.clamp_min(1.0)  # (M_j,)
            #print("r_tactic",r_tactic)
            """

            # --- tactic scalar advantage (first-token only) ---
            r_tactic = torch.zeros(M_j, device=It_b.device, dtype=It_b.dtype)

           
            r_sum = torch.zeros_like(r_tactic)
            r_sum.index_add_(0, ids_valid, adv_valid)
            r_tactic = r_sum



            # ------------------------------------------------------------------
            # 2) weights w_k = I_t / sum_{k in tactic m} I_t
            # ------------------------------------------------------------------



           

            tau=2.0

            if "softmax_tau" in self.divide_weight_method:
                # --- (a) τ-softmax :   exp(I/τ) / sum(exp(I/τ)) ---------------
                exp_I = (It_valid / tau).exp()  # (n,)
                denom = torch.zeros_like(r_tactic)
                denom.index_add_(0, ids_valid, exp_I)  # tactic별 sum
                denom = denom.clamp_min(eps)
                w_valid = exp_I / denom.index_select(0, ids_valid)  # (n,)
                #print("softmax")

            elif "sum" in self.divide_weight_method:
                denom = torch.zeros_like(r_tactic)
                denom.index_add_(0, ids_valid, It_valid)
                denom = denom.clamp_min(eps)
                w_valid = It_valid / denom.index_select(0, ids_valid)

            elif "max" in self.divide_weight_method:
                denom = torch.full_like(r_tactic, eps)
                denom.scatter_reduce_(0, ids_valid, It_valid,
                                      reduce='amax', include_self=True)
                w_valid = It_valid / denom.index_select(0, ids_valid)

            if self.advantage_distribute_top_k:
                keep_mask = torch.zeros_like(w_valid, dtype=torch.bool)

                # top-k 기준: entropy vs w_valid
                score_vec = It_valid

                for m in range(M_j):
                    idx_m = (ids_valid == m).nonzero(as_tuple=False).squeeze(-1)
                    if idx_m.numel() == 0:
                        continue
                    k = math.ceil(idx_m.numel() * 0.10)  # 
                    if k == 0:
                        continue
                    _, top_local = score_vec[idx_m].topk(k, largest=True)
                    keep_mask[idx_m[top_local]] = True

                if not self.reward_uniform_distribution:
                    
                    if "softmax_tau" in self.divide_weight_method:
                        raw_kept = exp_I.clone()  # 
                    else:
                        raw_kept = It_valid.clone()  # sum / max 

                    raw_kept[~keep_mask] = 0.0

                    sum_keep = torch.zeros(M_j, device=raw_kept.device)
                    sum_keep.index_add_(0, ids_valid[keep_mask], raw_kept[keep_mask])
                    sum_keep = sum_keep.clamp_min(eps)

                    w_valid = torch.zeros_like(raw_kept)
                    w_valid[keep_mask] = raw_kept[keep_mask] / sum_keep.index_select(0, ids_valid[keep_mask])
                    # ------------------------------------------

                if self.reward_uniform_distribution:
                    w_valid = torch.zeros_like(It_valid)
                    w_valid[keep_mask] = 1.0

            if self.first_tactic_token:
                first_token_mask = torch.ones_like(ids_valid, dtype=torch.bool)
                first_token_mask[1:] = ids_valid[1:] != ids_valid[:-1]        # True ↔ 

                w_valid[first_token_mask] = 1.0
            # ------------------------------------------------------------------
            # 3) token advantage = r_tactic[m] * w_k
            # ------------------------------------------------------------------
            #print("w_valid",w_valid)
            adv_token_valid = r_tactic.index_select(0, ids_valid) * w_valid

            # scatter back
            weights_out[b, tactic_tokens] = w_valid
            token_adv_out[b, tactic_tokens] = adv_token_valid




            if self.only_first_error_distribute:
                idx_all = torch.nonzero(tactic_tokens, as_tuple=False).squeeze(-1)  # (n,)
                # 
                first_valid = torch.ones_like(ids_valid, dtype=torch.bool)
                if ids_valid.numel() > 1:
                    first_valid[1:] = ids_valid[1:] != ids_valid[:-1]
                # 
                first_token_global = torch.zeros(L, dtype=torch.bool, device=ids_b.device)
                if idx_all.numel() > 0:
                    first_token_global[idx_all[first_valid]] = True

                # 
                fe_mask_b = all_first_error_masks[b].bool()
                comp_mask_b = completion_mask[b].bool()  # 완료 부분만 반영하고 싶으면 포함
                final_mask_b = (fe_mask_b | first_token_global) & comp_mask_b

                # 
                token_adv_out[b].mul_(final_mask_b.to(token_adv_out.dtype))


            torch.set_printoptions(precision=10, sci_mode=True)
            #print("token_adv_out[0]")
            #print(token_adv_out[0])



            #logging
            """
            mode = "eval" if self.control.should_evaluate else "train"
            # 2) changed token 
            eps = 1e-8
            changed_mask = (torch.abs(ratio - 1.0) > eps).float() * completion_mask.float()

            # 
            seq_lens = completion_mask.float().sum(dim=1)  # (B,)
            num_changed_per_s = changed_mask.sum(dim=1)  # (B,)

            # 3) 
            frac_changed_per_s = num_changed_per_s / seq_lens.clamp_min(1)  # (B,)

            # 3)
            self._metrics[mode]["ratio/mean"].append(frac_changed_per_s.mean().item())


            mean_ratio = ratio.mean().item()

            self._metrics[mode]["ratio/mean"].append(mean_ratio)
            """

        """"#debugging
        b = 0
        ids0 = tactic_ids[b].detach().cpu()
        r0 = tactic_token_adv[b].detach().cpu()
        adv0 = token_adv_out[b].detach().cpu()
        first_error_mask0 =all_first_error_masks[b].detach().cpu()
        
        print("\n[b=0] idx | id | tactic_token_adv | token_adv_out| first_error_mask")
        L = ids0.numel()
        for i in range(L):
            print(f"{i:4d} | {int(ids0[i].item()):3d} | {float(r0[i].item()): .8f} | {float(adv0[i].item()): .8f} | {float(first_error_mask0[i].item()): .8f}")
        """
        return token_adv_out, weights_out



    # @profiling_decorator
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
        adv_mask = inputs["adv_mask"]
        all_first_error_masks= inputs["all_first_error_masks"]
        all_tactic_ids= inputs["all_tactic_ids"]
        binary_reward = inputs["binary_reward"]

        # ref_per_token_logps = inputs["ref_per_token_logps"]

        #print("completion_ids.size",completion_ids.size())
        #print("completion_mask.size",completion_mask.size())
        per_token_logps,policy_entropies = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)





        """
        #log token-entropy pair
        comp_ids = completion_ids.detach().cpu()  # (B, L)
        comp_mask = completion_mask.detach().cpu().bool()
        comp_H = policy_entropies.detach().cpu()  # (B, L)

        # =====================================================================================
        # A. PRINT ──
        # =====================================================================================
        def decode_tokens(tokenizer, ids):
            return [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip()
                for tid in ids
            ]

        b = 0
        tok_str_b = decode_tokens(self.tokenizer, comp_ids[b].tolist())  # <- 수정

        print(f"\n[step {getattr(self, 'global_step', 0):,}] "
              f"Token-level entropy (batch {b})")
        for t, (tok, ent, m) in enumerate(zip(tok_str_b,
                                              comp_H[b].tolist(),
                                              comp_mask[b])):
            if not m:
                continue
            print(f"{t:3d}: {tok:>15s} | H = {ent:6.3f}")
        for t, (tok, ent, m) in enumerate(zip(tok_str_b,
                                              comp_H[b].tolist(),
                                              comp_mask[b])):
            if not m:  # padding
                continue
            print(f"{t:3d}: {tok:>15s} | H = {ent:6.3f}")

        # =====================================================================================
        # B. 
        # =====================================================================================
        csv_path = getattr(self, "_entropy_csv_path", "token_entropies.csv")
        set_header = not os.path.exists(csv_path)  

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if set_header:
                writer.writerow(
                    ["step", "batch", "token_idx", "token", "entropy"]
                )

            step = getattr(self, "global_step", 0)
            for b in range(comp_H.size(0)):
                tok_str_b = decode_tokens(self.tokenizer, comp_ids[b].tolist())  # <- 수정
                for t, (tok, ent, m) in enumerate(zip(tok_str_b,
                                                      comp_H[b].tolist(),
                                                      comp_mask[b])):
                    if m:
                        writer.writerow([step, b, t, tok, f"{ent:.6f}"])
        """








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

            reward = tactic_advantages + kl_reward  # why?

            if self.rloo_token_level:
                # print("num_batch",num_batch)
                # print("self.num_generations",self.num_generations)
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

            advantages = tactic_advantages
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
            old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
            #print("tactic_advantages",tactic_advantages)
            #print("binary_score", binary_score)


            """
            if self.tactic_distribution:
                # ──────────────────────────────────────────────────────────────
                # Token‑level weighting à‑la  |1 − π/π_old|   (see “ratio‑based
                # importance” in your note)
                # ──────────────────────────────────────────────────────────────

                # 1.  Per‑token probability ratio  πθ / πθ_old
                #     (torch.exp(log πθ − log πθ_old))
                ratio = torch.exp(per_token_logps - old_per_token_logps)  # (B, L)

                # 2.  Absolute deviation from 1  →  I_t  (indicator strength)
                I_t = torch.abs(1.0 - ratio)  # (B, L)

                # 3.  Mask‑out padding / non‑tactic tokens (keep zeros there)
                I_t = I_t * completion_mask  # (B, L)
                #   – If you have a more specific “tactic mask”, multiply by it
                #     instead / in addition to `completion_mask`.

                # 4.  Normalise *inside each sequence* so the weights
                #     for tokens in the same tactic step sum to 1.
                weights = I_t / (I_t.sum(dim=1, keepdim=True).clamp(min=1e-8))  # (B, L)

                # 5.  Distribute the tactic‑level advantages onto tokens.
                #     Detaching `weights` keeps the weighting constant w.r.t.
                #     the gradient of the current update (feel free to drop
                #     `.detach()` if you *want* it to back‑prop).
                tactic_advantages = tactic_advantages * weights.detach()
            """

            if self.tactic_distribution:
                token_adv, weights = self.distribute_tactic_reward_with_ids(
                    per_token_logps,
                    old_per_token_logps,
                    completion_mask,
                    all_tactic_ids,  # <── here it is
                    tactic_advantages,
                    policy_entropies,
                    all_first_error_masks
                )

                tactic_advantages= token_adv.detach()

            if self.asymmetric:

                eos_pos = completion_mask.sum(1).long() - 1  # (B,)

                prev = tactic_advantages.clone()

                tactic_advantages = (
                        tactic_advantages
                ).scatter(
                    dim=1,
                    index=eos_pos.unsqueeze(1),  # (B,1)
                    src=binary_score.unsqueeze(1)  # (B,1)
                )


                """
                torch.set_printoptions(precision=2, sci_mode=True)
                comparison = torch.stack([prev[0], tactic_advantages[0]], dim=-1)
                print("comparison (prev_tok0, prev_tok1, after_tok0, after_tok1):")
                print(comparison)
                """



                batch_mask = (binary_reward == 1)  # shape: (B,)
                batch_mask = batch_mask.unsqueeze(1)  # shape: (B, 1)
                #batch_mask = batch_mask.expand_as(tactic_advantages)  # shape: same as policy_entropies
                binary_score=binary_score.unsqueeze(1)
                raw_advantages = torch.where(
                    batch_mask,  # condition
                    binary_score,  # else (binary_reward==1)
                    tactic_advantages,  # if binary_reward==0

                )

                """
                torch.set_printoptions(precision=10, sci_mode=True)
                print("raw_advantages",raw_advantages)  # shape = (seq_len, 2)
                """


            else:
                if self.weighted_adv:
                    raw_advantages = self.alpha_advantage * tactic_advantages + (1-self.alpha_advantage)* binary_score.unsqueeze(1)
                    #raw_advantages =(1-self.alpha_advantage)* binary_score.unsqueeze(1)
                else:
                    raw_advantages = self.alpha_advantage * tactic_advantages +  binary_score.unsqueeze(1)

            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
            # _generate_and_score_completions) and use per_token_logps.detach() instead.

            # dropout
            # pass는 
            if self.negative_dropout:
                binary_mask = (binary_score != 0).clone()
                fail_idx = (binary_score == 0).nonzero(as_tuple=True)[0]
                num_keep = int(len(fail_idx) * (1 - self.dropout_rate))
                if num_keep > 0:
                    perm = torch.randperm(len(fail_idx), device=binary_score.device)
                    keep_idx = fail_idx[perm[:num_keep]]
                    binary_mask[keep_idx] = True

                # 토큰 단위 마스크
                token_mask = binary_mask.unsqueeze(1)

                advantages = advantages * token_mask



            if self.weighted_prob_advantage:
                is_tactic = all_tactic_ids.ge(0)  # [B, L]
                prev_ids = F.pad(all_tactic_ids, (1, 0), value=-10 ** 9)[:, :-1]  # [B, L]
                first_mask = is_tactic & (all_tactic_ids != prev_ids)  # [B, L]

                # 2) 
                first_logps = torch.where(first_mask, ref_per_token_logps, torch.zeros_like(ref_per_token_logps))
                num_first = first_mask.sum(dim=1).clamp(min=1)  # [B]
                ell_bar = first_logps.sum(dim=1) / num_first  # [B]

                # 3) 
                gamma = 0.2
                w_min, w_max = 1.0, 1.5

                fail = (binary_reward == 0)  # [B]
                num_fail = fail.sum()
                ell_fail = ell_bar[fail]  # 

                fail = (binary_reward == 0)
                if fail.any():
                    tau_fail = ell_bar[fail].median()
                    over = (ell_bar - tau_fail).clamp_min(0)
                    w = (1.0 + gamma * over).clamp(w_min, w_max)
                    w = torch.where(fail, w, torch.ones_like(w))  # 
                else:
                    w = torch.ones_like(ell_bar)  # 


                # 4) 
                scale = torch.where(first_mask, w.unsqueeze(1), torch.ones_like(tactic_advantages))  # [B, L]
                raw_advantages = (tactic_advantages * scale).detach()

                """"#debugging
                with torch.no_grad():
                    b = 0
                    ids0 = all_tactic_ids[b].detach().cpu()  # [L]
                    first_mask0 = first_mask[b].detach().cpu()  # [L] (bool)
                    tadv0 = tactic_advantages[b].detach().cpu()  # [L]
                    scale0 = scale[b].detach().cpu()  # [L]
                    raw_adv0 = raw_advantages[b].detach().cpu()  # [L]
                    print("\n[b=0] idx |  id | first | tactic_adv     | scale         | raw_adv")
                    L = ids0.numel()
                    for i in range(L):
                        _id = int(ids0[i].item())
                        _fst = int(first_mask0[i].item())  # bool -> 0/1
                        _tadv = float(tadv0[i].item())
                        _scl = float(scale0[i].item())
                        _raw = float(raw_adv0[i].item())
                        print(f"{i:4d} | {_id:3d} | {_fst:5d} | {_tadv: .8f} | {_scl: .8f} | {_raw: .8f}")
                """
            # advantages = masked_whiten(raw_advantages, completion_mask, shift_mean=True)
            advantages = raw_advantages


            epsilon_high = self.epsilon + 0.08
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + epsilon_high)

            if self.delta_clip is not None:
                coef_1 = torch.clamp(coef_1, max=self.delta_clip)

            #print("tactic_advantages",tactic_advantages[0])


            if self.entropy_adv:
                batch_mask = (binary_reward == 0).unsqueeze(1).expand_as(policy_entropies)
                # (2)
                final_mask = batch_mask & adv_mask.bool()

                # (3) 
                ent_selected = self.entropy_coef*(torch.where(final_mask, policy_entropies, torch.zeros_like(policy_entropies)))


                """
                torch.set_printoptions(precision=10, sci_mode=True)
                pairs = torch.stack([tactic_advantages[0], ent_selected[0]], dim=-1)
                print("pairs",pairs)  # shape = (seq_len, 2)
                """

                #print(ent_selected[0])
                advantages=(advantages+ent_selected).detach()

            if self.entropy_reg:
                batch_mask = (binary_reward == 0).unsqueeze(1).expand_as(policy_entropies)
                # (2) 
                final_mask = batch_mask & adv_mask.bool()

                # (3) entropy 
                ent_selected = self.entropy_coef * (
                    torch.where(final_mask, policy_entropies, torch.zeros_like(policy_entropies)))

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)



            if self.entropy_reg:
                if self.beta != 0.0:
                    per_token_loss = per_token_loss + self.beta * per_token_kl - self.entropy_coef*ent_selected
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()


            else:
                if self.beta != 0.0:
                    per_token_loss = per_token_loss + self.beta * per_token_kl
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            """
            print(
                f"loss      = {loss}\n",
                f"advantages      = {advantages}\n",
                f"per_token_logps      = {per_token_logps}\n"
                f"old_per_token_logps     = {old_per_token_logps}",
            flush = True)
            """

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = ((coef_1 < 1 - self.epsilon) & (advantages.unsqueeze(1) < 0)) | (
                (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        )

        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        masked_entropy = policy_entropies * completion_mask
        if self.entropy_adv or self.entropy_reg:
            masked_selected_entropy =ent_selected* completion_mask
            mean_selected_entropy_per_seq = masked_selected_entropy.sum(-1) / token_counts
            self._metrics[mode]["policy_selected_entrpoy"].append(
            self.accelerator.gather_for_metrics(mean_selected_entropy_per_seq).mean().item())

        token_counts = completion_mask.sum(-1).clamp(min=1)
        mean_entropy_per_seq = masked_entropy.sum(-1) / token_counts  # (B,)
        mean_entropy_per_seq = masked_entropy.sum(-1) / token_counts  # (B,)
        self._metrics[mode]["policy_entrpoy"].append(self.accelerator.gather_for_metrics(mean_entropy_per_seq).mean().item())


        """
        sum_local = ell_fail.sum()
        cnt_local = torch.tensor([ell_fail .numel()], device=ell_fail.device, dtype=torch.long)

        sum_all = self.accelerator.gather_for_metrics(sum_local)  # [num_procs]
        cnt_all = self.accelerator.gather_for_metrics(cnt_local)  # [num_procs]

        fail_tactic_prob = (sum_all.sum() / cnt_all.sum().clamp(min=1)).item()

        self._metrics[mode]["fail_tactic_prob"].append(fail_tactic_prob)

        """


        # print("self._metrics[mode]",self._metrics[mode])
        # print("loss",loss)
        return loss
