import json
import gc
import os
import sys
from copy import deepcopy
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
import itertools
import fcntl

import torch
from torch.distributed import _functional_collectives as funcol

import torch._inductor.config
import torch._dynamo.config

import heapq
import datetime
from tqdm import tqdm, trange
from lean_dojo import *
import subprocess
from transformers import AutoModelForCausalLM


import vllm
import transformers

def _load_model(checkpoint_path, device, precision, tp_size):
    model_name = checkpoint_path
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype='bfloat16',
        max_num_batched_tokens=32768,
        trust_remote_code=True,
        enforce_eager=True
    )
    #model_name = "/localdata_ssd/Lean/checkpoints/internlm/internlm2-math-base-7b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('<|im_start|>' + message['role'] +
                   '\n' + message['content'])
        if i + 1 != total_step:
            result += '<|im_end|>\n'
        elif message['role'] == 'user':
            result += '<|im_end|>\n<|im_start|>assistant\n'
    return result


def _prompt_fewshot(state):
    prompt = f"My LEAN 4 state is:\n```lean\n" + state + \
             "```\nPlease write down the reasoning that leads to the possible next tactic and then predict the tactic to help me prove the theorem."
    # "```\nPlease predict a possible tactic to help me prove the theorem."
    prompt = [{"role": "user", "content": prompt}]
    return chat_template_to_prompt(prompt)


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


@torch.no_grad()
def generate_tactic(
        prompts,
        model,
        tokenizer,
        max_seq_len,
        num_samples,
        temperature
):
    texts = []
    prompts = [_prompt_fewshot(prompt) for prompt in prompts]
    params = vllm.SamplingParams(
        n=1,
        temperature=temperature,
        use_beam_search=False,
        max_tokens=max_seq_len,
        stop=["<|im_end|>"]
    )
    outputs = model.generate(prompts, params, use_tqdm=False)
    for i in range(len(prompts)):
        output = outputs[i].outputs[0]
        text = output.text.replace(tokenizer.eos_token, '')
        texts.append(text)

    return texts

def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        batch_size,
        timeout=600,
        early_stop=False,
        max_seq_len=512,
        top_k=200
) -> dict:
    """Best first search."""
    attempt_results = []
    print("theorem: ", theorem)
    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            start = time.time()
            proof_finished = False
            cnt = 0
            states, steps, traces = [], [], []
            for i in range(num_samples):
                states.append(init_state)
                steps.append([])
                traces.append([])

            for iteration in trange(max_iters):
                istart = time.time()
                if istart - start > timeout:
                    break
                if proof_finished:
                    break

                ts = [_tactic_state(state) for state in states]

                step_cands = generate_tactic(
                    ts,
                    model,
                    tokenizer,
                    max_seq_len=max_seq_len,
                    num_samples=1,
                    temperature=temperatures
                )

                # if iteration < 2:
                #    print(iteration, " # state: ",ts[0])
                #    print("tatics: ", step_cands[0])

                step_cots = step_cands
                step_cands = [s.split("```lean\n")[-1].split('```')[0].split('---')[0].strip() for s in step_cands]
                # print(step_cands[:10])
                for i in range(num_samples):
                    state, step, step_cot = states[i], step_cands[i], step_cots[i]
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        "tactic": step,
                        "full_cot": step_cot,
                        "state_before": _tactic_state(state)
                    }
                    if isinstance(result, ProofFinished):
                        attempt_results.append({
                            'theorem': theorem.full_name,
                            'proof': steps[i] + [step],
                            'success': True,
                            'failure_reason': '',
                            'trace': traces[i] + [step_trace],
                            'temperature': temperatures,
                            'elapsed': start - time.time(),
                            'iteration': iteration
                        })
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        break
                    elif isinstance(result, TacticState):
                        # if _tactic_state(result) not in visited:
                        # Score is negative log probability summed across steps
                        # new_score = (total_score - score)
                        cnt += 1
                        states[i] = result
                        steps[i].append(step)
                        traces[i].append(step_trace)
    except (DojoInitError, DojoHardTimeoutError, DojoCrashError) as e:
        print("Error: ", e)
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'SearchEnded'
        })

    return attempt_results


def _save(model_name, results, args_dict, output_dir, shard):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        'results__%s__%s.json' % (model_name.replace('/', '_'), shard)
    )
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'args': args_dict
        }, f, indent=4)
        print(output_file)


def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                # assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    elif 'leandojo' in dataset_name:
        with open(dataset_path) as f:
            data = json.load(f)
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data


def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")


def resume_from(results_filename, data):
    results = json.load(open(results_filename))['results']
    data = data[len(results):]
    print("=== Resuming from %d" % (len(results)))
    return results, data


def make_output_dir(output_dir):
    dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def main(
    seed: int,
    batch_size: int = 4,
    num_samples: int = 5,
    temperature: float = 0.8,
    checkpoint_path = None ,
    compile: bool = True,
    default_compile: bool = False,
    finetune_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_prefix: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    on_the_fly_8bit_quantization: bool = False,
    args = None
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    tp_size = 1
    if use_tp:
        tp_size = tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model, tokenizer = _load_model(checkpoint_path, device, precision, tp_size)

    if on_the_fly_8bit_quantization:
        print("Quantizing model ...")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime_on_the_fly()
        model = model.to(device=device)
        model = model.eval()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    dp_rank = 0  # get_data_parallel_rank()
    tp_rank = 0  # get_model_parallel_rank()

    dp_size = 0  # get_data_parallel_world_size()

    # if tp_rank == 0:
    #    output_writer = open(output_file, "a")

    batch_idx = 0

    gc.collect()
    torch.cuda.empty_cache()

    # if compile:
    #    remove_all_backward_hooks(model)

    output_dir = make_output_dir(args.output_dir)

    repo, data = _load_data(args.dataset_name, args.dataset_path)
    shard_size = len(data) // args.num_shards
    # import random
    # random.seed(1926)
    # random.shuffle(data)
    data = data[args.shard * shard_size:(args.shard + 1) * shard_size] if args.num_shards > 1 + args.shard else data[
                                                                                                                args.shard * shard_size:]
    # data = data[(1690+850+1440+1000+800):]
    # data = data[(9000):]
    print("Shard size: %d" % (len(data)))

    start = time.time()
    results = []
    for example in tqdm(data, total=len(data)):
        file_path = example['file_path']
        theorem_name = example['full_name']
        theorem = Theorem(repo, file_path, theorem_name)
        for _ in range(4):
            attempt_results = best_first_search(
                theorem, model, tokenizer,
                max_iters=args.max_iters,
                temperatures=temperature,
                num_samples=args.num_samples,
                batch_size=batch_size,
                timeout=args.timeout,
                early_stop=args.early_stop,
                top_k=args.top_k
            )
            if any([x['success'] for x in attempt_results]):
                break

        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }

        results.append(result)

        _save(
            model_name="internLM-7b-math",
            results=results,
            args_dict={},
            output_dir=output_dir,
            shard=args.shard
        )
        print_stats(results)
        # The proof search occasionally leaves Lean processes open. As a workaround,
        # we periodically kill all Lean processes. Note that this may cause a proof search failure.
        if args.shard == 0:
            hours = 60 * 60 * args.clear_process_hours
            if time.time() - start > hours:
                print("=== Killing active leanprover processes to mitigate leak")
                os.system("ps aux | grep leanprover | awk '{print $2}' | xargs kill -9")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--seed", type=int, default=1926, help="Random seed for reproducibility."
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--default_compile",
        action="store_true",
        help="Whether to compile the model with default settings.",
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune checkpoint path.",
    )

    parser.add_argument(
        "--finetune_checkpoint_prefix",
        type=str,
        default=None,
        help="Finetune checkpoint prefix.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Size of tensor parallelism.",
    )

    parser.add_argument(
        "--on_the_fly_8bit_quantization",
        action="store_true",
        help="Whether to quantize after loading the model.",
    )


    parser.add_argument(
        '--dataset-name',
        default='minif2f-test',
        choices=['minif2f-valid', 'minif2f-test', 'leandojo']
    )

    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shard-base', type=int, required=True)
    parser.add_argument('--dataset-path', default='data/minif2f.jsonl')
    parser.add_argument('--output-dir', default='output/minif2f')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--num-examples', type=int, default=-1)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--clear-process-hours', type=int, default=15)
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument('--local-rank', type=int, default=None)
    args = parser.parse_args()
    args.shard = args.shard - args.shard_base
    main(
        args.seed,
        args.batch_size,
        args.num_samples,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.default_compile,
        args.finetune_checkpoint_path,
        args.finetune_checkpoint_prefix,
        args.tensor_parallel_size,
        args.on_the_fly_8bit_quantization,
        args
    )