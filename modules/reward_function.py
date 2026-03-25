import random
import warnings
from copy import deepcopy
import gc
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Dict, Union, Tuple
from .prover.lean.verifier import Lean4ServerScheduler
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
from easydict import EasyDict as AttrDict
import random
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from statistics import mean as stat_mean, pvariance
from collections import defaultdict
from typing import List,Tuple
Interval = Tuple[int, int, int]


def compute_line_offsets(text: str):
    """
    Return (line_offsets, lines) for a multiline text.
    line_offsets[i] = the absolute char index in text where line i (0-based) starts.
    lines is the list of lines including their trailing newlines (except possibly last line).
    """
    lines = text.splitlines(keepends=True)
    line_offsets = []
    running_offset = 0
    for line_str in lines:
        line_offsets.append(running_offset)
        running_offset += len(line_str)
    return line_offsets, lines


def mark_char_scores(char_scores, full_text, error, data):
    """
    Mark -1 in char_scores for each error range described by (pos, endPos).
    Each 'pos' is a dict with 'line' and 'column' (1-based).
    """
    # print("data",data)
    line_offsets, _ = compute_line_offsets(full_text)
    for idx, item in enumerate(data):
        pos = item.get("pos")  # or None if missing
        end_pos = item.get("endPos")

        # If both are missing, do nothing.
        if not pos and not end_pos:
            continue

        # Figure out start and end lines/columns.
        # If `endPos` is missing, treat it as the same as `pos`.
        # If `pos` is missing, treat it as the same as `endPos`.
        start_line = pos["line"] if pos else end_pos["line"]
        start_col = pos["column"] if pos else end_pos["column"]
        end_line = end_pos["line"] if end_pos else pos["line"]
        end_col = end_pos["column"] if end_pos else pos["column"]

        # Compute absolute offsets for start and end.
        # Make sure lines exist in line_offsets, or guard with boundary checks.
        start_abs = line_offsets[start_line - 1] + (start_col - 1)
        end_abs = line_offsets[end_line - 1] + (end_col)

        # Mark the scores
        is_error = (item.get("severity") == "error")
        score_to_mark = -1 if is_error else 1
        # print("is_error",is_error)
        for i in range(start_abs, end_abs):
            if 0 <= i < len(char_scores):
                char_scores[i] = score_to_mark


def mark_char_scores_snippet(char_scores, snippet_text, data, default_to_error=False):
    """
    Mark +1 or -1 in 'char_scores' for each dict in 'data'.
    Here, 'snippet_text' is *just the code snippet*, so line references
    are (1-based) within that snippet.
    """
    line_offsets, _ = compute_line_offsets(snippet_text)

    for item in data:
        pos = item.get("pos")
        end_pos = item.get("endPos")
        if not pos and not end_pos:
            continue
        # If either is missing, treat them as the same
        if not pos:
            pos = end_pos
        if not end_pos:
            end_pos = pos

        start_line = pos["line"]
        start_col = pos["column"]
        end_line = end_pos["line"]
        end_col = end_pos["column"]

        # The snippet_text is the entire domain, so no line_offset needed
        start_line_idx = (start_line - 1)
        end_line_idx = (end_line - 1)

        # Boundary checks
        if not (0 <= start_line_idx < len(line_offsets)):
            continue
        if not (0 <= end_line_idx < len(line_offsets)):
            continue

        start_abs = line_offsets[start_line_idx] + (start_col - 1)
        end_abs = line_offsets[end_line_idx] + end_col

        # If "severity" is "error" or we forced default_to_error => -1
        # Tactics => +1
        is_error = (item.get("severity") == "error") or default_to_error
        score_to_mark = -1 if is_error else 1

        # Fill char_scores in snippet
        for i in range(start_abs, end_abs):
            if 0 <= i < len(char_scores):
                char_scores[i] = score_to_mark


# treebased
def build_interval_tree(intervals: List[Interval]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build a containment tree.
    Returns (roots, flat_list_of_all_nodes).
    Each node is a dict with keys: start, end, label, children, return.
    """
    # Sort by start asc, end desc so that parents appear before their children
    sorted_intv = sorted(intervals, key=lambda x: (x[0], -x[1]))
    stack = []  # holds the current path of open intervals
    nodes = []  # flat list of every node

    for s, e, lbl in sorted_intv:
        node = {"start": s, "end": e, "label": lbl, "children": []}
        # climb up until we find a parent that contains (s,e)
        while stack and not (stack[-1]["start"] <= s and stack[-1]["end"] >= e):
            stack.pop()
        if stack:  # stack[-1] is the parent
            stack[-1]["children"].append(node)
        stack.append(node)
        nodes.append(node)

    # Roots are those that never became anyone’s child
    roots = [n for n in nodes if all(n not in p["children"] for p in nodes)]
    return roots, nodes

def build_potential_interval_tree(intervals: List[Interval]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build a containment tree.
    Returns (roots, flat_list_of_all_nodes).
    Each node is a dict with keys: start, end, label, children, return.
    """
    # Sort by start asc, end desc so that parents appear before their children
    sorted_intv = sorted(intervals, key=lambda x: (x[0], -x[1]))
    stack = []  # holds the current path of open intervals
    nodes = []  # flat list of every node

    for s, e, lbl,val,pot in sorted_intv:
        node = {"start": s, "end": e, "label": lbl,"real_val":val, "potential":pot, "children": []}
        # climb up until we find a parent that contains (s,e)
        while stack and not (stack[-1]["start"] <= s and stack[-1]["end"] >= e):
            stack.pop()
        if stack:  # stack[-1] is the parent
            stack[-1]["children"].append(node)
        stack.append(node)
        nodes.append(node)

    # Roots are those that never became anyone’s child
    roots = [n for n in nodes if all(n not in p["children"] for p in nodes)]
    return roots, nodes


def _dfs_compute_returns(node: Dict, gamma: float) -> float:
    """Post‑order DFS that fills node['return'] and returns it."""
    child_ret_sum = sum(_dfs_compute_returns(c, gamma) for c in node["children"])
    node_return = node["label"] + gamma * child_ret_sum
    node["return"] = node_return
    return node_return


def compute_tactic_scores_for_output_deepseek(
        prompts,
        completions,
        outputs_list,
        extracted_codes,
        tokenizer
):
    """

    Returns: all_token_scores, all_token_texts
    """
    all_token_scores = []
    all_token_texts = []
    binary_pass_score = []
    for i, (prompt, completion, out_data, snippet) in enumerate(
            zip(prompts, completions, outputs_list, extracted_codes)
    ):
        full_text = prompt + completion

        # Build a big pos_scores array for the entire full_text
        pos_scores = [0] * len(full_text)

        # (A) Score the snippet by line/column
        snippet_pos_scores = [0] * len(snippet)

        # Mark snippet's tactics => +1
        if "tactics" in out_data and out_data["tactics"]:
            mark_char_scores_snippet(
                snippet_pos_scores,
                snippet,
                out_data["tactics"],
                default_to_error=False
            )

        # Mark snippet's errors => -1
        if "errors" in out_data and out_data["errors"]:
            mark_char_scores_snippet(
                snippet_pos_scores,
                snippet,
                out_data["errors"],
                default_to_error=True
            )

        # (B) Find snippet in the full_text, so we can map snippet scores
        snippet_index = full_text.find(snippet)
        if snippet_index != -1:
            for idx in range(len(snippet)):
                # Copy snippet_pos_scores into pos_scores
                if 0 <= snippet_index + idx < len(pos_scores):
                    # If snippet_pos_scores[idx] == -1, that overrides
                    # a +1. If pos_scores is already -1, keep it -1, etc.
                    # We'll do: error overrides tactic if both exist.
                    if pos_scores[snippet_index + idx] == -1:
                        continue
                    pos_scores[snippet_index + idx] = snippet_pos_scores[idx]

        else:

            pass

        # (C) Tokenize full_text and compute averages
        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        token_scores = []
        token_texts = []
        prompt_len = len(prompt)

        # We only keep tokens fully in the completion portion => start >= prompt_len
        for tid, (start, end) in zip(input_ids, offsets):
            if start >= prompt_len:
                slice_scores = pos_scores[start:end]
                if slice_scores:
                    avg_score = sum(slice_scores) / len(slice_scores)
                else:
                    avg_score = 0.0

                token_str = full_text[start:end]
                token_scores.append(avg_score)
                token_texts.append(token_str)

        if out_data['complete'] == 'True':
            binary_pass_score.append(1)
        else:
            binary_pass_score.append(0)



        all_token_scores.append(token_scores)
        all_token_texts.append(token_texts)

    return all_token_scores, all_token_texts, binary_pass_score


def compute_tactic_scores_for_output(prompts, completions, outputs_list, tokenizer):  # token-level reward
    """
 

    """
    all_token_scores = []
    all_token_texts = []  # If you want to keep track of the actual token strings

    for i, (prompt, completion, out_data) in enumerate(zip(prompts, completions, outputs_list)):
        full_text = prompt + completion

        # 1) Build pos_scores (per-character)
        pos_scores = [0] * len(full_text)

        # 2) give 1 to all tactics
        if len(out_data.get("tactics", [])) > 0:
            mark_char_scores(pos_scores, full_text, False, out_data["tactics"])

        # 3) If there are errors, mark -1
        if len(out_data.get("errors", [])) > 0:
            mark_char_scores(pos_scores, full_text, True, out_data["errors"])
        # print("pos_scores",pos_scores)
        # 4) Tokenize full_text
        #    Make sure you use a fast tokenizer with offset mappings
        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]  # list of (start, end)

        # 5) For each token, compute the average of the char scores in [start, end)
        token_scores = []
        token_texts = []
        prompt_len = len(prompt)
        token_id = []
        # if model outputs with BOS token

        for tid, (start, end) in zip(input_ids, offsets):  # character-token alignment
            # If you only want tokens fully in the completion portion,
            # check if start >= prompt_len
            # (If you want partial coverage for tokens that straddle the boundary,
            # you'd have to do a partial average or something more advanced.)
            if start >= prompt_len:  # if model outputs with BOS token
                # sum / len for the slice
                slice_scores = pos_scores[start:end]
                if len(slice_scores) > 0:
                    avg_score = sum(slice_scores) / len(slice_scores)
                else:
                    avg_score = 0  # edge case if start==end

                # The raw text for this token
                token_str = full_text[start:end]

                token_scores.append(avg_score)
                token_texts.append(token_str)
                token_id.append(tid)
        # print("index",i)
        # print("scoring_completion_token_id", token_id)
        # print("scoring_completion_token_score", token_scores)
        all_token_scores.append(token_scores)
        all_token_texts.append(token_texts)

        # print("all_token_scores",all_token_scores)
        # print("all_token_texts",all_token_texts)
    return all_token_scores, all_token_texts


def build_tactic_tree(full_text: str, tactics: list) -> list:
    """
    Given a flat list of tactics, build a parent–child hierarchy based on
    strictly nested start/end positions. Return a list of 'root' tactics
    (those not nested inside any other).

    Each tactic is a dict with:
      {
         "pos":    {"line": int, "column": int},
         "endPos": {"line": int, "column": int},
         ...
      }
    We add:
      tactic["abs_start"] = absolute start offset in the combined text
      tactic["abs_end"]   = absolute end offset
      tactic["children"]  = []  # if not present
    """
    line_offsets, _ = compute_line_offsets(full_text)

    # 1) Assign absolute positions
    for t in tactics:
        sL, sC = t["pos"]["line"], t["pos"]["column"]
        eL, eC = t["endPos"]["line"], t["endPos"]["column"]
        abs_start = line_offsets[sL - 1] + (sC - 1)
        abs_end = line_offsets[eL - 1] + eC
        t["abs_start"] = abs_start
        t["abs_end"] = abs_end
        if "children" not in t:
            t["children"] = []

    # 2) Sort tactics by ascending abs_start (if tie, by ascending abs_end)
    tactics.sort(key=lambda x: (x["abs_start"], x["abs_end"]))

    # 3) We'll keep a stack of currently open (parent) tactics
    stack = []

    for t in tactics:
        # Pop from stack until the top of the stack actually contains t
        while stack:
            top = stack[-1]
            # Check if top *contains* t
            if (top["abs_start"] <= t["abs_start"] and
                    t["abs_end"] <= top["abs_end"]):
                # T is indeed contained in top => we found our parent
                break
            else:
                # T is not contained by the top => pop
                stack.pop()

        # If there's anything left in the stack, the new top is the parent
        if stack:
            stack[-1]["children"].append(t)

        # Push the current tactic on the stack
        stack.append(t)

    # 4) The 'root' tactics are those never added as a child
    #    So we can find them by checking any tactic that isn't in some parent's .children
    #    or simply build a set of all children and subtract from the full set
    all_children = []
    for t in tactics:
        all_children.extend(t["children"])

    roots = [t for t in tactics if t not in all_children]
    return roots


def gather_intervals_no_split(snippet_text, out_data):
    """
    Return a list of intervals (startAbs, endAbs, label) in snippet_text coordinates.
    Tactics => +1, Errors => 0.
    """
    intervals = []
    line_offsets, _ = compute_line_offsets(snippet_text)

    def to_abs(pos):
        if not pos:
            return None
        line_idx = pos["line"] - 1
        col_idx = pos["column"] - 1
        if line_idx < 0 or line_idx >= len(line_offsets):
            return None
        return line_offsets[line_idx] + col_idx

    for t in out_data.get("tactics", []):
        start = to_abs(t.get("pos") or t.get("endPos"))
        end = to_abs(t.get("endPos") or t.get("pos"))
        if start is not None and end is not None and end > start:
            intervals.append((start, end, 1))
    for e in out_data.get("errors", []):
        start = to_abs(e.get("pos") or e.get("endPos"))
        end = to_abs(e.get("endPos") or e.get("pos"))
        if start is not None and end is not None and end > start:
            intervals.append((start, end, 0))
    return intervals


def convert_snippet_intervals_to_full_text(prompt, completion, snippet, intervals):
    """
    Convert intervals in snippet coordinates into full-text coordinates.
    """
    full_text = prompt + completion
    snippet_index = full_text.find(snippet)
    if snippet_index == -1:
        return []
    abs_intervals = []
    for (start_snip, end_snip, label) in intervals:
        abs_start = snippet_index + start_snip
        abs_end = snippet_index + end_snip
        if 0 <= abs_start < len(full_text) and 0 < abs_end <= len(full_text):
            abs_intervals.append((abs_start, abs_end, label))
        # if label==1:
        # print("abs_start",abs_start)
        # print("abs_end",abs_end)
    return abs_intervals

    """
    def mark_char_scores_snippet(char_scores, snippet_text, data, default_to_error=False):

        #Mark +1 or -1 in 'char_scores' for each dict in 'data' using snippet_text.

        line_offsets, _ = compute_line_offsets(snippet_text)
        for item in data:
            pos = item.get("pos")
            end_pos = item.get("endPos")
            if not pos and not end_pos:
                continue
            if not pos:
                pos = end_pos
            if not end_pos:
                end_pos = pos
            start_line = pos["line"]
            start_col = pos["column"]
            end_line = end_pos["line"]
            end_col = end_pos["column"]
            start_line_idx = start_line - 1
            end_line_idx = end_line - 1
            if not (0 <= start_line_idx < len(line_offsets)):
                continue
            if not (0 <= end_line_idx < len(line_offsets)):
                continue
            start_abs = line_offsets[start_line_idx] + (start_col - 1)
            end_abs = line_offsets[end_line_idx] + end_col
            is_error = (item.get("severity") == "error") or default_to_error
            score_to_mark = -1 if is_error else 1
            for i in range(start_abs, end_abs):
                if 0 <= i < len(char_scores):
                    char_scores[i] = score_to_mark
    """


def deduplicate_intervals(intervals):
    if not intervals:
        return []

    # (start, end) 기준으로 정렬 (label은 정렬 순서에 영향을 주지 않음)
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    deduped = []
    i = 0
    while i < len(intervals):
        start, end, label = intervals[i]
        # 동일한 (start, end)를 가지는 interval들을 candidates에 모음.
        candidates = [(start, end, label)]
        j = i + 1
        while j < len(intervals) and intervals[j][0] == start and intervals[j][1] == end:
            candidates.append(intervals[j])
            j += 1
        # candidates 중 하나라도 label이 -1이면, -1을 유지.
        if any(c[2] == -1 for c in candidates):
            deduped.append((start, end, -1))
        elif any(c[2] == 0 for c in candidates):
            deduped.append((start, end, 0))
        else:
            deduped.append(candidates[0])
        i = j
    return deduped


def compute_returns_tree(
        full_text: str,
        intervals: List[Interval],
        tokenizer,

        gamma: float = 0.9,
        prompt_len: int = 0,
):
    """
    Like compute_returns_no_split, but reward is propagated along the
    containment tree instead of a flat, left‑to‑right timeline.
    """
    # 1. deduplicate & build the tree ----------------------------------------
    intervals = deduplicate_intervals(intervals)
    roots, all_nodes = build_interval_tree(intervals)

    # 2. tree‑discounted returns --------------------------------------------
    for r in roots:
        _dfs_compute_returns(r, gamma)

    # 3. baseline & advantages ----------------------------------------------

    returns = [n["return"] for n in all_nodes]
    baseline = sum(returns) / len(
        returns) if returns else 0.0  # baseline = sum(returns) / len(returns) if returns else 0.0
    # 4. packaging -----------------------------------------------------------
    results = []
    if not returns:  # <- nothing to do
        return ([],)  # (empty results, baseline 0)

    m = stat_mean(returns)

    var = pvariance(returns)

    std = math.sqrt(var + 1e-04)

    for n in all_nodes:
        results.append(
            {
                "start": n["start"],
                "end": n["end"],
                "label": n["label"],
                "immediate_reward": n["label"],
                "discounted_tree_return": n["return"],
                "advantage": (n["return"] - m) / std,
            }
        )

    # (Optional) sort so +1 labels first, then by start
    results.sort(key=lambda x: (0 if x["label"] == 1 else 1, x["start"]))

    return (results,)


def compute_returns_no_split(full_text, intervals, tokenizer, gamma=0.9, prompt_len=0):
    """
    Given full_text and a list of absolute intervals (start, end, label),
    assign a fixed immediate reward = label for each interval,
    discount them backward, and compute baseline and advantages.
    """
    intervals = deduplicate_intervals(intervals)
    intervals = sorted(intervals, key=lambda x: x[0])
    # For fixed reward, each interval's reward is simply its label.
    rewards = [label for (start, end, label) in intervals]
    discounted = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted.insert(0, running)
        print("running ", type(running))

    #return normalized
    mean = stat_mean(discounted)
    var = pvariance(discounted)


    std = math.sqrt(var + 1e-04)

    # 4)  advantage
    advantages = [(d - mean) / std for d in discounted]



    results = []
    for (start, end, label), ret, adv in zip(intervals, discounted, advantages):
        results.append({
            "start": start,
            "end": end,
            "label": label,
            "immediate_reward": label,
            "discounted_return": ret,
            "advantage": adv
        })
    results = sorted(results, key=lambda x: (0 if x["label"] == 1 else 1, x["start"]))
    return results,


def assign_advantage_to_tokens(full_text, offsets, snippet, intervals_info, prompt_len,delta2):
    """
    Given a list of intervals_info (each with keys "start", "end", "label"),
    assign the full advantage (here using the label as the advantage value)
    to any token whose span (given by offsets) overlaps the interval,
    but only if the token is in the completion portion (i.e. its start >= prompt_len).

    If a token overlaps multiple intervals, here we assign the advantage of the first matching interval.
    """

    character_adv = [0] * len(full_text)
    character_mask = [0] * len(full_text)
    character_tid = [-1] * len(full_text)
    first_delta2_mask = [0] * len(full_text)

    seen_first_delta2 = False

    # (B) Find snippet in the full_text, so we can map snippet scores
    snippet_index = full_text.find(snippet)
    if snippet_index != -1:
        for t_id,info in enumerate(intervals_info):
            int_start = info["start"]
            int_end = info["end"]
            ent_start= info["entropy_start"]
            ent_end= info["entropy_end"]
            try:
                adv = info["advantage"]  # or use info["advantage"] if computed separately  #discounted_return, label
            except:
                adv = info["advantage"]

            label = info.get("label", 0.0)
            character_adv[int_start:int_end + 1] = [adv] * (int_end - int_start + 1)
            character_mask[ent_start:ent_end + 1] = [1] * (ent_end - ent_start + 1)
            character_tid[ent_start:ent_end + 1] = [t_id] * (ent_end - ent_start + 1)

            """
            is_delta2 = False
            if delta2 is not None:
                is_delta2 = (abs(label - delta2) <= 1e-9)

            """

    else:
        # If snippet wasn't found, we just won't mark anything.
        # Or you could log a warning, etc.
        # print("no matching")
        pass

    return character_adv,character_mask,character_tid,first_delta2_mask


def compute_token_level_advantages(prompts, completions, outputs_list, tokenizer, extracted_codes, type, gamma):
    """
    Process each sample (prompt, completion, out_data, extracted_code) individually.
    Returns lists (one per sample) of token_scores, token_texts, token_advantages, intervals_info, and baseline.
    """
    all_token_scores = []
    all_token_texts = []
    binary_pass_score = []

    for prompt, completion, out_data, snippet in zip(prompts, completions, outputs_list, extracted_codes):
        full_text = prompt + completion
        # Use the provided snippet (ensure it's a string)
        prompt_len = len(prompt)
        if not isinstance(snippet, str):
            continue  # or raise an error
        # 1. Gather intervals from out_data (in snippet coordinates)

        if type == 'advantage':
            snippet_intervals = gather_intervals_no_split(snippet,
                                                          out_data)  # get the position of tactic and error in the extracted_code ex) error1=(1,10, -1), tactic1= (26,57,1)
            # 2. (Optionally merge tactic intervals; here we assume no splitting is needed)
            # For simplicity, we assume out_data gives the intervals correctly.
            abs_intervals = convert_snippet_intervals_to_full_text(prompt, completion, snippet,
                                                                   snippet_intervals)  # get the position of tactic and error in the full text (prompt+completion)

            intervals_info, = compute_returns_no_split(full_text, abs_intervals, tokenizer, gamma=gamma,
                                                       prompt_len=prompt_len)

        if type == 'tree':
            snippet_intervals = gather_intervals_no_split(snippet,
                                                          out_data)  # get the position of tactic and error in the extracted_code ex) error1=(1,10, -1), tactic1= (26,57,1)
            # 2. (Optionally merge tactic intervals; here we assume no splitting is needed)
            # For simplicity, we assume out_data gives the intervals correctly.
            abs_intervals = convert_snippet_intervals_to_full_text(prompt, completion, snippet,
                                                                   snippet_intervals)  # get the position of tactic and error in the full text (prompt+completion)

            intervals_info, = compute_returns_tree(full_text, abs_intervals, tokenizer, gamma=gamma,
                                                   prompt_len=prompt_len)
            # if len(intervals_info)==0:
            #    print("no interval",out_data)
            #    print("no interbval snippet", snippet)
            # 3. Tokenize full_text
            # print("intervals_info", intervals_info)
        encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
        input_ids = encoded["input_ids"]
        # 4. Build pos_scores at character level
        pos_scores = [0] * len(full_text)
        snippet_pos_scores = [0] * len(snippet)

        token_scores = []
        token_texts = []

        if type == 'reward':
            if "tactics" in out_data and out_data["tactics"]:
                mark_char_scores_snippet(snippet_pos_scores, snippet, out_data["tactics"], default_to_error=False)
            if "errors" in out_data and out_data["errors"]:
                mark_char_scores_snippet(snippet_pos_scores, snippet, out_data["errors"], default_to_error=True)
            # 5. Compute token-level average score (only for tokens in completion)

            # 6. compute token level reward
            snippet_index = full_text.find(snippet)
            if snippet_index != -1:
                for idx in range(len(snippet)):
                    # Copy snippet_pos_scores into pos_scores
                    if 0 <= snippet_index + idx < len(pos_scores):
                        # If snippet_pos_scores[idx] == -1, that overrides
                        # a +1. If pos_scores is already -1, keep it -1, etc.
                        # We'll do: error overrides tactic if both exist.
                        if pos_scores[snippet_index + idx] == -1:
                            continue
                        pos_scores[snippet_index + idx] = snippet_pos_scores[idx]

            else:
                # If snippet wasn't found, we just won't mark anything.
                # Or you could log a warning, etc.
                # print("no matching")
                pass

        # 7. Assign advantages to tokens based on intervals_info
        elif type == 'advantage' or type == 'tree':
            character_advantages = assign_advantage_to_tokens(full_text, offsets, snippet, intervals_info, prompt_len)

        for tid, (start, end) in zip(input_ids, offsets):
            if start >= prompt_len:
                if type == 'reward':
                    slice_scores = pos_scores[start:end]
                    avg = sum(slice_scores) / len(slice_scores) if slice_scores else 0.0
                    token_scores.append(avg)


                elif type == 'advantage' or type == 'tree':
                    slice_adv_scores = character_advantages[start:end]
                    avg = sum(slice_adv_scores) / len(slice_adv_scores) if slice_adv_scores else 0.0
                    token_scores.append(avg)
                token_texts.append(full_text[start:end])

        if out_data['complete'] == 'True':
            binary_pass_score.append(1)
        else:
            binary_pass_score.append(0)

        all_token_scores.append(token_scores)
        all_token_texts.append(token_texts)

    return all_token_scores, all_token_texts, binary_pass_score,





##calculate advantage, grouped_mean_baseline










#score as value, change base reward 1, delta, delta_2

def delta_deduplicate_intervals(intervals,delta2):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    deduped = []
    i = 0
    while i < len(intervals):
        start, end, label = intervals[i]

        candidates = [(start, end, label)]
        j = i + 1
        while j < len(intervals) and intervals[j][0] == start and intervals[j][1] == end:
            candidates.append(intervals[j])
            j += 1
        # candidates 중 하나라도 label이 -1이면, -1을 유지.
        if any(c[2] == delta2 for c in candidates):
            deduped.append((start, end, delta2))
        else:
            deduped.append(candidates[0])
        i = j
    return deduped


def delta_gather_intervals_no_split(
        snippet_text, out_data,
        pass_score, delta0, delta1, delta2,
        first_error: bool
):
    """
    Return a list of (start, end, label) in *snippet_text* coordinates.
      • pass_score == 1 → original behaviour
      • pass_score != 1 and first_error == True → everything from the first
        error onward is labelled delta2
      • pass_score != 1 and first_error == False → original delta1 / delta2 mix
    """
    # -------------------------------------------------- common setup
    intervals = []
    line_offsets, _ = compute_line_offsets(snippet_text)

    def to_abs(pos):
        if not pos:
            return None
        line_idx = pos["line"] - 1
        col_idx  = pos["column"] - 1
        if line_idx < 0 or line_idx >= len(line_offsets):
            return None
        return line_offsets[line_idx] + col_idx
    # -------------------------------------------------- logic branches
    if not first_error:
        if pass_score == 1:
            # original “passed” behaviour
            for t in out_data.get("tactics", []):
                start = to_abs(t.get("pos") or t.get("endPos"))
                end   = to_abs(t.get("endPos") or t.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, 1))
            for e in out_data.get("errors", []):
                start = to_abs(e.get("pos") or e.get("endPos"))
                end   = to_abs(e.get("endPos") or e.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, 0))
        else:
            # original “failed but keep delta1/delta2 distinction” behaviour
            for t in out_data.get("tactics", []):
                start = to_abs(t.get("pos") or t.get("endPos"))
                end   = to_abs(t.get("endPos") or t.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, delta1))
            for e in out_data.get("errors", []):
                start = to_abs(e.get("pos") or e.get("endPos"))
                end   = to_abs(e.get("endPos") or e.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, delta2))
    else:
        if pass_score == 1:
            # original “passed” behaviour
            for t in out_data.get("tactics", []):
                start = to_abs(t.get("pos") or t.get("endPos"))
                end   = to_abs(t.get("endPos") or t.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, delta0))
            for e in out_data.get("errors", []):
                start = to_abs(e.get("pos") or e.get("endPos"))
                end   = to_abs(e.get("endPos") or e.get("pos"))
                if start is not None and end is not None and end > start:
                    intervals.append((start, end, 0))

        else:
            raw = []
            first_error_pos = None

            for e in out_data.get("errors", []):
                start = to_abs(e.get("pos") or e.get("endPos"))
                end   = to_abs(e.get("endPos") or e.get("pos"))
                if start is not None and end is not None and end > start:
                    raw.append((start, end, 'error'))
                    if first_error_pos is None or start < first_error_pos:
                        first_error_pos = start

            for t in out_data.get("tactics", []):
                start = to_abs(t.get("pos") or t.get("endPos"))
                end   = to_abs(t.get("endPos") or t.get("pos"))
                if start is not None and end is not None and end > start:
                    raw.append((start, end, 'tactic'))

            raw.sort(key=lambda x: x[0])  # by start
            for start, end, kind in raw:
                if first_error_pos is not None and start >= first_error_pos:
                    intervals.append((start, end, delta2))
                else:
                    intervals.append((start, end, delta2 if kind == 'error' else delta1))

    return intervals


def delta_value_compute_returns_no_split(full_text, intervals, tokenizer, mean,ad_baseline,delta1, delta2, gamma=0.9, prompt_len=0, adv_method=None,rloo_mean=None):
    """
    Given full_text and a list of absolute intervals (start, end, label),
    assign a fixed immediate reward = label for each interval,
    discount them backward, and compute baseline and advantages.
    """
    intervals = delta_deduplicate_intervals(intervals,delta2)
    intervals = sorted(intervals, key=lambda x: x[0])
    # For fixed reward, each interval's reward is simply its label.
    rewards = [label for (start, end, label) in intervals]
    discounted = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted.insert(0, running)
        #print("running ", type(running))

    if 'seq' in ad_baseline.lower():        #seq는 tactic score의 mean
        baseline= np.mean(rewards)
    elif 'group' in ad_baseline.lower():       #group은 group에서의 biniary score의 mean
        baseline=mean
    elif 'zero' in ad_baseline.lower():
        baseline=0
    elif 'rloo' in ad_baseline.lower():
        baseline=rloo_mean

    advantages = [ d - baseline for d in discounted]



    results = []
    for (start, end, label), ret, adv in zip(intervals, discounted, advantages):
        results.append({
            "start": start,
            "end": end,
            "label": label,
            "immediate_reward": label,
            "discounted_return": ret,
            "advantage": adv
        })
    o = {1: 0} if any(r['label'] == 1 for r in results) else {delta1: 0, delta2: 1}
    results.sort(key=lambda x: (o.get(x['label'], len(o)), x['start']))

    return results,

def compute_potentials(
    deduped: List[Tuple[Any, Any, Any]],
    delta2,
    gamma: float = 0.9,
    potent_type= None,
    potent_positive=None,
    pass_score=None,
    shift_potential=None,
    potent_coef=None
) -> List[Tuple[Any, Any, Any, float, float]]:
    """
    Return a list where each interval tuple is extended with:
        (start, end, label, value, potential)

    value      = v_i  = -distance/n,  where distance is
                • first_delta2_idx - i  if delta2 exists and i < first_delta2_idx
                • 0                     if i >= first_delta2_idx
                • (n - i)               if no delta2 exists
    potential  = F_i = (v_{i+1} - v_i) * (1 + gamma*(n - i - 1))
                with v_{n} := 0 for the last element.

    Parameters
    ----------
    deduped : list of (start, end, label)
              Must already be sorted and de‑duplicated.
    delta2  : the special label that stops the distance count.
    gamma   : slope‑scaling factor (default 0.1).
    """
    n = len(deduped)
    if n == 0:
        return []

    # locate first delta2
    try:
        first_delta_idx = next(i for i, (_, _, lbl) in enumerate(deduped) if lbl == delta2)
        has_delta2 = True
    except StopIteration:
        first_delta_idx = None
        has_delta2 = False

    # pre‑compute value (v) for each index
    values: List[float] = []
    if potent_positive:
        for i in range(n):
            if pass_score == 1:
                if has_delta2:
                    dist = max(first_delta_idx - i, 0)
                else:
                    dist = n - i
                if 'distance_to_goal' in potent_type:
                    values.append(-dist / n)
                elif 'monotone_growth' in potent_type:
                    values.append(-(dist / n) ** (0.5))
                elif 'normalized_exponential' in potent_type:
                    lam = 3.0  # config knob
                    denom = 1.0 - math.exp(-lam)
                    p = dist / n  # progress value in [0,1]
                    phi = -(1.0 - math.exp(-lam * p)) / denom  # Φ(p)
                    values.append(phi)
                elif "exponential_step" in potent_type:
                    step= i+1
                    num_step=n
                    phi=math.exp(step-num_step)
                    values.append(phi)
            else:
                values.append(0)


    else:
        for i in range(n):
            if has_delta2:
                dist = max(first_delta_idx - i, 0)
            else:
                dist = n - i
            if 'distance_to_goal' in potent_type:
                values.append(-dist / n)
            elif 'monotone_growth' in potent_type:
                values.append(-(dist / n)**(0.5))
            elif 'normalized_exponential' in potent_type:
                lam = 3.0  #config knob
                denom = 1.0 - math.exp(-lam)
                p = dist / n  # progress value in [0,1]
                phi = -(1.0 - math.exp(-lam  * p)) / denom  # Φ(p)
                values.append(phi)
            elif "exponential_step" in potent_type:
                step = i + 1
                num_step = n
                if deduped[i][2]==delta2:
                    phi=-math.exp(step - num_step)
                else:
                    phi = math.exp(step - num_step)
                values.append(phi)
            elif "proof_step" in potent_type:
                step = i + 1
                num_step = n
                phi=step/num_step
                if pass_score == 1:
                    values.append(phi)
                else:
                    values.append(-phi)



    if 'monotone_growth' in potent_type:
        gamma=1

    # compute potentials
    potentials: List[float] = []
    if shift_potential:
        for i in range(n):
            v_now = values[i]
            v_prev = values[i - 1] if i > 0 else 0.0
            diff = gamma * v_now - v_prev
            potentials.append(potent_coef*diff)


    else:
        for i in range(n):
            v_now = values[i]
            v_next = values[i + 1] if i < n - 1 else 0.0
            diff = gamma *v_next - v_now
            #coef = 1 + gamma * (n - i - 1)
            potentials.append(potent_coef*diff)

    # build extended tuples
    extended = [
        (s, e, lbl, val, pot)
        for (s, e, lbl), val, pot in zip(deduped, values, potentials)
    ]
    #print("extended_node",extended )
    return extended


def compute_returns_seq(nodes, gamma: float = 0.9):
    """

    """
    # start 기준으로 정렬
    nodes = sorted(nodes, key=lambda n: n["start"])

    G = 0.0
    # 뒤에서 앞으로 누적
    for n in reversed(nodes):
        G = n["label"] + gamma * G
        n["label"] = G
        n["return"] = G

    return nodes

def delta_value_compute_returns_tree(
        full_text: str,
        intervals: List[Interval],
        tokenizer,
        mean,
        ad_baseline,
        delta0,
        delta1,
        delta2,
        pass_score,
        gamma: float = 0.9,
        prompt_len: int = 0,
        score_assign= "last",
        adv_method=None,
        rloo_mean=None,
        potent_func=None,
        potent_type=None,
        potent_positive=None,
        shift_potential=None,
        potent_coef=None,
        entropy_position=None
):
    """
    Like compute_returns_no_split, but reward is propagated along the
    containment tree instead of a flat, left‑to‑right timeline.
    """
    # 1. deduplicate & build the tree ----------------------------------------
    intervals = delta_deduplicate_intervals(intervals,delta2)
    if potent_func:
        intervals= compute_potentials(intervals,delta2,gamma,potent_type,potent_positive,pass_score,shift_potential,potent_coef)
        roots, all_nodes = build_potential_interval_tree(intervals)


    else:
        roots, all_nodes = build_interval_tree(intervals)
    #print("value_reward")


    # 2. tree‑discounted returns --------------------------------------------
    for r in roots:
        _dfs_compute_returns(r, gamma)

    #####!!!SHOULD DELETE THIS RETURN FUNCTION!!!!!!
    #all_nodes= compute_returns_seq(all_nodes,gamma)
    # 3. baseline & advantages ----------------------------------------------

    values = [n["label"] for n in all_nodes]
    results = []
    if not values:  # <- nothing to do
        return [],0  # (empty results, baseline 0)

    if 'seq' in ad_baseline.lower():        #seq는 tactic score의 mean (first trial)
        baseline= np.mean(values)
    elif 'group' in ad_baseline.lower():       #group은 group에서의 biniary score의 mean
        baseline=mean
        #print("baseline!!",baseline)
    elif 'zero' in ad_baseline.lower():
        baseline=0
    elif 'rloo' in ad_baseline.lower():
        baseline=rloo_mean



    if "label" in adv_method.lower():
        if 'last' in score_assign.lower():
            if potent_func:
                for n in all_nodes:
                    results.append(
                        {
                            "start": n["end"],
                            "end": n["end"],
                            "label": n["label"],
                            "immediate_reward": n["label"],
                            "discounted_tree_return": n["return"],
                            "advantage": n["label"] + n["potential"]- baseline,
                        }
                    )
            else:
                for n in all_nodes:
                    results.append(
                        {
                            "start": n["end"],
                            "end": n["end"],
                            "label": n["label"],
                            "immediate_reward": n["label"],
                            "discounted_tree_return": n["return"],
                            "advantage": n["label"]  - baseline,

                        }
                    )
        if 'all' in score_assign.lower():
            for n in all_nodes:
                # 1) entropy 위치에 따라 ent_start/ent_end 결정
                if entropy_position.lower() == 'all':
                    ent_start, ent_end = n['start'], n['end']
                elif entropy_position.lower() == 'start':
                    ent_start = ent_end = n['start']
                else:
                    ent_start = ent_end = None

                # 2) advantage 계산 (potent_func 여부에 따라)
                if potent_func:
                    adv = n['label'] + n['potential'] - baseline
                else:
                    adv = n['label'] - baseline

                # 3) 결과 append (end만 n['end'] 로 다름)
                results.append({
                    "start": n["start"],
                    "end": n["end"],
                    "entropy_start": ent_start,
                    "entropy_end": ent_end,
                    "label": n["label"],
                    "immediate_reward": n["label"],
                    "discounted_tree_return": n["return"],
                    "advantage": adv,
                })
        elif 'start' in score_assign.lower():
            for n in all_nodes:
                # entropy 위치에 따라 분기
                if entropy_position.lower() == 'all':
                    ent_start = n['start']
                    ent_end = n['end']
                elif entropy_position.lower() == 'start':
                    ent_start = n['start']
                    ent_end = n['start']
                else:
                    # 필요하다면 다른 옵션 처리
                    ent_start = None
                    ent_end = None

                # advantage 계산
                if potent_func:
                    adv = n['label'] + n['potential'] - baseline
                else:
                    adv = n['label'] - baseline

                """ delta0 as zero
                if n['label']==delta0:
                    adv=0
                """
                results.append({
                    "start": n["start"],
                    "end": n["start"],
                    "entropy_start": ent_start,
                    "entropy_end": ent_end,
                    "label": n["label"],
                    "immediate_reward": n["label"],
                    "discounted_tree_return": n["return"],
                    "advantage": adv,
                })

        elif 'mix' in score_assign.lower():
            if pass_score==1:
                if potent_func:
                    for n in all_nodes:
                        results.append(
                            {
                                "start": n["start"],
                                "end": n["end"],
                                "label": n["label"],
                                "immediate_reward": n["label"],
                                "discounted_tree_return": n["return"],
                                "advantage": n["label"] + n["potential"] - baseline,

                            }
                        )
                else:
                    for n in all_nodes:
                        results.append(
                            {
                                "start": n["start"],
                                "end": n["end"],
                                "label": n["label"],
                                "immediate_reward": n["label"],
                                "discounted_tree_return": n["return"],
                                "advantage": n["label"] - baseline,

                            }
                        )
            else:
                if potent_func:
                    for n in all_nodes:
                        results.append(
                            {
                                "start": n["end"],
                                "end": n["end"],
                                "label": n["label"],
                                "immediate_reward": n["label"],
                                "discounted_tree_return": n["return"],
                                "advantage": n["label"] + n["potential"] - baseline,

                            }
                        )
                else:
                    for n in all_nodes:
                        results.append(
                            {
                                "start": n["end"],
                                "end": n["end"],
                                "label": n["label"],
                                "immediate_reward": n["label"],
                                "discounted_tree_return": n["return"],
                                "advantage": n["label"] - baseline,

                            }
                        )


        o = {1: 0} if any(r['label'] == 1 for r in results) else {delta1: 0, delta2: 1}
        results.sort(key=lambda x: (o.get(x['label'], len(o)), x['start']))
        mean_state_score = sum(r["label"] for r in results) / len(results)

    else:
        if 'last' in score_assign.lower():
            for n in all_nodes:
                results.append(
                    {
                        "start": n["end"],
                        "end": n["end"],
                        "label": n["label"],
                        "immediate_reward": n["label"],
                        "discounted_tree_return": n["return"],
                        "advantage": n["return"] - baseline,
                    }
                )
        else:
            for n in all_nodes:
                results.append(
                    {
                        "start": n["start"],
                        "end": n["end"],
                        "label": n["label"],
                        "immediate_reward": n["label"],
                        "discounted_tree_return": n["return"],
                        "advantage": n["return"] - baseline,
                    }
                )

        o = {1: 0} if any(r['label'] == 1 for r in results) else {delta1: 0, delta2: 1}
        results.sort(key=lambda x: (o.get(x['label'], len(o)), x['start']))
        mean_state_score = sum(r["return"] for r in results) / len(results)





    return results,mean_state_score


def fail_aware_compute_token_level_advantages(prompts, completions, outputs_list, tokenizer, extracted_codes, type, gamma,num_generation,delta0,delta1,delta2, ad_baseline, score_assign,adv_method,first_error,potent_func,potent_type,potent_positive,shift_potential,potent_coef,entropy_position):
    """
    Process each sample (prompt, completion, out_data, extracted_code) individually.
    Returns lists (one per sample) of token_scores, token_texts, token_advantages, intervals_info, and baseline.
    """

    ad_baseline= ad_baseline
    all_token_scores = []
    all_token_texts = []
    all_adv_masks = []
    all_first_error_masks=[]
    all_token_ids = []
    all_tactic_ids=[]
    all_intervals_info = []

    binary_pass_score = [float(item["complete"]) for item in outputs_list]

    n = len(binary_pass_score)
    num_groups = math.ceil(n / num_generation)  # works even if it doesn't divide evenly

    group_means = [
        float(np.mean(binary_pass_score[g * num_generation: (g + 1) * num_generation]))
        for g in range(num_groups)
    ]

    #total = sum(binary_pass_score)
    #loo_means = [(total - s) / (n - 1) for s in binary_pass_score]
    rloo_means = []  # will match `binary_pass_score` in length

    for g in range(num_groups):
        start = g * num_generation
        end = min((g + 1) * num_generation, n)  # protect the last (possibly shorter) group
        group = binary_pass_score[start:end]

        k = len(group)
        group_sum = sum(group)

        # leave-one-out mean for every element in this group
        rloo_means.extend(
            (group_sum - x) / (k - 1) if k > 1 else float("nan")  # NaN if the group has only 1 item
            for x in group
        )

    tactic_mean = []
    for idx,(prompt, completion, out_data, snippet) in enumerate(zip(prompts, completions, outputs_list, extracted_codes)):
        pass_score = binary_pass_score[idx]
        baseline = group_means[idx // num_generation]
        rloo_mean=rloo_means[idx]
        #print("baseline",baseline)
        full_text = prompt + completion
        # Use the provided snippet (ensure it's a string)
        prompt_len = len(prompt)
        if not isinstance(snippet, str):
            continue  # or raise an error
        # 1. Gather intervals from out_data (in snippet coordinates)

        if type == 'advantage':
            snippet_intervals = delta_gather_intervals_no_split(snippet,
                                                          out_data,pass_score,delta0,delta1,delta2,first_error)  # get the position of tactic and error in the extracted_code ex) error1=(1,10, -1), tactic1= (26,57,1)
            # 2. (Optionally merge tactic intervals; here we assume no splitting is needed)
            # For simplicity, we assume out_data gives the intervals correctly.
            abs_intervals = convert_snippet_intervals_to_full_text(prompt, completion, snippet,
                                                                   snippet_intervals)  # get the position of tactic and error in the full text (prompt+completion)

            intervals_info, = delta_value_compute_returns_no_split(full_text, abs_intervals, tokenizer, baseline,ad_baseline,delta0,delta1, delta2,gamma=gamma,
                                                       prompt_len=prompt_len,adv_method=adv_method,rloo_mean=rloo_mean)

        if type == 'tree':
            snippet_intervals = delta_gather_intervals_no_split(snippet,
                                                          out_data,pass_score, delta0, delta1, delta2,first_error)  # get the position of tactic and error in the extracted_code ex) error1=(1,10, -1), tactic1= (26,57,1)
            # 2. (Optionally merge tactic intervals; here we assume no splitting is needed)
            # For simplicity, we assume out_data gives the intervals correctly.
            abs_intervals = convert_snippet_intervals_to_full_text(prompt, completion, snippet,
                                                                   snippet_intervals)  # get the position of tactic and error in the full text (prompt+completion)

            intervals_info,mean_state_score = delta_value_compute_returns_tree(full_text, abs_intervals, tokenizer,baseline, ad_baseline,delta0,delta1, delta2,pass_score, gamma=gamma,
                                                   prompt_len=prompt_len,score_assign=score_assign,adv_method=adv_method,rloo_mean=rloo_mean,potent_func=potent_func,potent_type=potent_type,potent_positive=potent_positive,shift_potential=shift_potential,potent_coef=potent_coef,entropy_position=entropy_position)
            # if len(intervals_info)==0:
            #    print("no interval",out_data)
            #    print("no interbval snippet", snippet)
            # 3. Tokenize full_text
            # print("intervals_info", intervals_info)
        tactic_mean.append(mean_state_score)
        encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
        input_ids = encoded["input_ids"]
        # 4. Build pos_scores at character level
        pos_scores = [0] * len(full_text)
        snippet_pos_scores = [0] * len(snippet)

        token_scores = []
        token_masks=[]
        first_error_token_masks=[]
        token_texts = []
        token_id = []
        token_tactic_ids=[]

        if type == 'reward':
            if "tactics" in out_data and out_data["tactics"]:
                mark_char_scores_snippet(snippet_pos_scores, snippet, out_data["tactics"], default_to_error=False)
            if "errors" in out_data and out_data["errors"]:
                mark_char_scores_snippet(snippet_pos_scores, snippet, out_data["errors"], default_to_error=True)
            # 5. Compute token-level average score (only for tokens in completion)

            # 6. compute token level reward
            snippet_index = full_text.find(snippet)
            if snippet_index != -1:
                for idx in range(len(snippet)):
                    # Copy snippet_pos_scores into pos_scores
                    if 0 <= snippet_index + idx < len(pos_scores):
                        # If snippet_pos_scores[idx] == -1, that overrides
                        # a +1. If pos_scores is already -1, keep it -1, etc.
                        # We'll do: error overrides tactic if both exist.
                        if pos_scores[snippet_index + idx] == -1:
                            continue
                        pos_scores[snippet_index + idx] = snippet_pos_scores[idx]

            else:
                # If snippet wasn't found, we just won't mark anything.
                # Or you could log a warning, etc.
                # print("no matching")
                pass

        # 7. Assign advantages to tokens based on intervals_info
        elif type == 'advantage' or type == 'tree':
            character_advantages, character_mask,character_tid,first_error_mask = assign_advantage_to_tokens(full_text, offsets, snippet, intervals_info, prompt_len,delta2)

        for tid, (start, end) in zip(input_ids, offsets):
            if start >= prompt_len:
                if type == 'reward':
                    slice_scores = pos_scores[start:end]
                    avg = sum(slice_scores) / len(slice_scores) if slice_scores else 0.0
                    token_scores.append(avg)


                elif type == 'advantage' or type == 'tree':
                    slice_adv_scores = character_advantages[start:end]
                    if 'last' in score_assign.lower():
                        avg = slice_adv_scores[-1]  if slice_adv_scores else 0.0
                    elif 'start' in score_assign.lower():
                        avg = slice_adv_scores[0]  if slice_adv_scores else 0.0
                    elif 'all' in score_assign.lower():
                        avg = sum(slice_adv_scores) / len(slice_adv_scores) if slice_adv_scores else 0.0
                    elif 'mix' in score_assign.lower():
                        if pass_score==1:
                            avg = sum(slice_adv_scores) / len(slice_adv_scores) if slice_adv_scores else 0.0
                        else:
                            avg = slice_adv_scores[-1] if slice_adv_scores else 0.0
                    token_scores.append(avg)




                slice_msk = character_mask[start:end]
                assigned = 1 if any(slice_msk) else 0
                token_masks.append(assigned)

                first_error_slice_msk = first_error_mask[start:end]
                first_error_assigned = 1 if any(first_error_slice_msk) else 0
                first_error_token_masks.append(first_error_assigned )


                slice_tids = character_tid[start:end]
                # pick the most frequent id in that slice (ignores -1)
                valid_tids = [t for t in slice_tids if t >= 0]
                t_id = max(valid_tids, key=valid_tids.count) if valid_tids else -1
                token_tactic_ids.append(t_id)


                token_texts.append(full_text[start:end])
                token_id.append(tid)
        if len(token_id)<1024:
            token_scores.append(0)
            token_id.append(100001)
            token_texts.append("eos")
            token_masks.append(0)
            first_error_token_masks.append(0)
            token_tactic_ids.append(-1)
            #print("completion_ids_in_rewarwd_function",token_id)
        #print("token_scores_in_rewarwd_function",token_scores)
        all_token_scores.append(token_scores)
        all_token_texts.append(token_texts)
        all_token_ids.append(token_id)
        all_adv_masks.append(token_masks)
        all_first_error_masks.append(first_error_token_masks)
        all_tactic_ids.append(token_tactic_ids)
        all_intervals_info.append(intervals_info)
    tactic_mean=sum(tactic_mean)/len(tactic_mean)

    """"#Debugging

    """


    return all_token_scores, all_token_texts, binary_pass_score, all_token_ids,tactic_mean,all_adv_masks,all_tactic_ids,all_first_error_masks







def list_of_lists_to_padded_tensor(list_of_lists, padding_value=0):
    """
    Convert list of variable-length lists to a padded 2D Tensor
    shape: (batch_size, max_length_in_batch)
    """
    #for i,seq in enumerate(list_of_lists):
    #    print("index",i)
    #    print("len(seq)",len(seq))
    max_len = max(len(seq) for seq in list_of_lists) if list_of_lists else 0
    batch_size = len(list_of_lists)
    padded_tensor = torch.full((batch_size, max_len), fill_value=padding_value, dtype=torch.float)

    for i, seq in enumerate(list_of_lists):
        length = len(seq)
        padded_tensor[i, :length] = torch.tensor(seq, dtype=torch.float)

    return padded_tensor

def rloo_list_of_lists_to_padded_tensor(list_of_lists, max_len, padding_value=0):
    """
    Convert list of variable-length lists to a padded 2D Tensor
    shape: (batch_size, max_length_in_batch)
    """
    #for i,seq in enumerate(list_of_lists):
    #    print("index",i)
    #    print("len(seq)",len(seq))
    batch_max_len = max(len(seq) for seq in list_of_lists) if list_of_lists else 0
    if max_len == batch_max_len:
        max_len = batch_max_len
        batch_size = len(list_of_lists)
        padded_tensor = torch.full((batch_size, max_len), fill_value=padding_value, dtype=torch.float)

        for i, seq in enumerate(list_of_lists):
            length = len(seq)
            padded_tensor[i, :length] = torch.tensor(seq, dtype=torch.float)
    else:
        batch_size = len(list_of_lists)
        padded_tensor = torch.full((batch_size, max_len), fill_value=padding_value, dtype=torch.float)

        for i, seq in enumerate(list_of_lists):
            # truncate if too long
            length = min(len(seq), max_len)
            # convert & slice to max_len, then assign
            padded_tensor[i, :length] = torch.tensor(seq[:length], dtype=torch.float)

    return padded_tensor


def extract_code(inputs):
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except:
        return "None"

def extract_code_stp(inputs):
    try:
        return re.search(r'```lean4\n(?s)(.*)', inputs, re.DOTALL).group(1)
    except:
        return "None"



def lean4_value_reward(prompts, completions, processing_class,num_generation):      #lean score as value
    texts = [p + c for p, c in zip(prompts, completions)]

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15,  memory_limit=10, name='verifier',extra_args=AttrDict(allTactics=True))

    extracted_code=[extract_code_stp(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    #extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    #Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    #As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(random.choice(outputs_list))

    all_token_scores, all_token_texts, binary_pass_score  = fail_aware_compute_token_level_advantages(
        prompts, completions, outputs_list,processing_class,extracted_code, "tree",0.9,num_generation,-0.05,-0.1,ad_baseline="group")
    #value_compute_token_level_advantages ,grouped_compute_token_level_advantages


    # 3. Convert to a padded tensor if desired
    #    Each row in padded_scores corresponds to one (prompt+completion) example
    #    The columns are the tokens in the completion portion
    padded_scores = list_of_lists_to_padded_tensor(all_token_scores, padding_value=0)



    binary_score = [float(item["complete"]) for item in outputs_list]






    """
    n = len(binary_score)
    num_groups = math.ceil(n / num_generation)  # works even if it doesn't divide evenly

    group_means = [
        float(np.mean(binary_score[g * num_generation: (g + 1) * num_generation]))
        for g in range(num_groups)
    ]



    for i, (scores, texts) in enumerate(zip(all_token_scores, all_token_texts)):
        print(f"--- Completion #{i} ---")
        if i ==0:
            baseline = group_means[0 // num_generation]
            print("baseline",baseline)
            print("output_list",outputs_list[0])
            print("completions",completions[0])
            for token_str, sc in zip(texts, scores):
                print(f"Token '{token_str}' => Score {sc:.2f}")


    """




    #print("padded_scores",padded_scores.size())
    lean4_scheduler.close()
    return padded_scores ,binary_score




def lean4_grpo_reward(prompts, completions, **kwargs):
    texts = [p + c for p, c in zip(prompts, completions)]
    #print("prompts",prompts)
    #print("completions",completions)
    #print("texts1:",texts)
    #print("type",type(texts[0]))
    #print("\n\n")
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15,  memory_limit=10, name='verifier',extra_args=AttrDict(allTactics=True))
    #print("texts2:", texts)
    extracted_code=[extract_code_stp(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    #extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    #Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    #As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    #print(random.choice(outputs_list))
    binary_score = [float(item["complete"]) for item in outputs_list]
    lean4_scheduler.close()
    return binary_score



def deepseek_lean4_grpo_reward (prompts, completions, **kwargs):
    texts = [p + c for p, c in zip(prompts, completions)]

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15,  memory_limit=10, name='verifier',extra_args=AttrDict(allTactics=True))
    #print("texts2:", texts)
    extracted_code=[extract_code(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    #extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    #Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    #As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(random.choice(outputs_list))
    #print(random.choice(outputs_list))
    binary_score = [float(item["complete"]) for item in outputs_list]
    lean4_scheduler.close()
    return binary_score


def lean4_rloo_reward(texts, **kwargs):

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15,  memory_limit=10, name='verifier',extra_args=AttrDict(allTactics=True))
    #print("texts2:", texts)
    extracted_code=[extract_code_stp(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    #extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    #Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    #As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(random.choice(outputs_list))
    binary_score = [float(item["complete"]) for item in outputs_list]
    lean4_scheduler.close()
    return binary_score



def lean4_rloo_custom_reward(prompts, completions, processing_class,max_len,num_generation,delta0, delta1, delta2 ,adv_baseline, score_assign ,adv_method, parse_method,first_error,potent_func,potent_type,potent_positive,shift_potential,potent_coef,entropy_position):
    texts = [p + c for p, c in zip(prompts, completions)]

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15, memory_limit=10, name='verifier',
                                           extra_args=AttrDict(allTactics=True))
    # print("texts2:", texts)
    extracted_code = [extract_code_stp(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    # extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    # Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    # As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(random.choice(outputs_list))
    # print("output_list",outputs_list)
    # print("rewarding start")
    all_token_scores, all_token_texts, binary_pass_score,all_token_ids,tactic_mean,all_adv_masks,all_tactic_ids,all_first_error_masks = fail_aware_compute_token_level_advantages(
        prompts, completions, outputs_list,processing_class,extracted_code, parse_method,0.9,num_generation,delta0,delta1,delta2,ad_baseline=adv_baseline, score_assign=score_assign,adv_method=adv_method, first_error=first_error,potent_func=potent_func,potent_type=potent_type,potent_positive=potent_positive,shift_potential=shift_potential,potent_coef=potent_coef,entropy_position=entropy_position)
    #value_compute_token_level_advantages ,grouped_compute_token_level_advantages

    # 3. Convert to a padded tensor if desired
    #    Each row in padded_scores corresponds to one (prompt+completion) example
    #    The columns are the tokens in the completion portion
    padded_scores = rloo_list_of_lists_to_padded_tensor(all_token_scores,max_len,padding_value=0)
    all_adv_masks= rloo_list_of_lists_to_padded_tensor(all_adv_masks,max_len,padding_value=0)
    all_tactic_ids=rloo_list_of_lists_to_padded_tensor(all_tactic_ids,max_len,padding_value=-1)
    all_first_error_masks=rloo_list_of_lists_to_padded_tensor(all_first_error_masks,max_len,padding_value=0)
    binary_score = [float(item["complete"]) for item in outputs_list]
    #print("padded_scores",padded_scores.size())
    lean4_scheduler.close()

    timeout_flags = [
        1.0 if "TimeoutExpired" in (item.get("system_errors") or "") else 0.0
        for item in outputs_list
    ]


    """
    n = len(binary_score)
    num_groups = math.ceil(n / num_generation)  # works even if it doesn't divide evenly

    group_means = [
        float(np.mean(binary_score[g * num_generation: (g + 1) * num_generation]))
        for g in range(num_groups)
    ]



    for i, (scores, texts, ids,tactic_ids) in enumerate(zip(all_token_scores, all_token_texts,all_token_ids,all_tactic_ids)):
        print(f"--- Completion #{i} ---")
        if i ==0:
            baseline = group_means[0 // num_generation]
            print("baseline",baseline)
            print("output_list",outputs_list[0])
            print("completions",completions[0])
            for idx, (token_str, sc, id,tactic_ids) in enumerate(zip(texts, scores,ids,tactic_ids)):
                print(f"{idx} Token '{token_str} Token_id {id}' => Score {sc:.2f}, Tactic_ids {tactic_ids}")
    """




    return padded_scores, binary_score,all_token_ids,tactic_mean,all_adv_masks,all_tactic_ids,all_first_error_masks,timeout_flags




def deepseek_lean4_rloo_custom_reward(prompts, completions, processing_class,max_len,num_generation,delta0, delta1, delta2 ,adv_baseline, score_assign ,adv_method, parse_method,first_error,potent_func,potent_type,potent_positive,shift_potential,potent_coef,entropy_position):
    texts = [p + c for p, c in zip(prompts, completions)]

    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=15, memory_limit=10, name='verifier',
                                           extra_args=AttrDict(allTactics=True))
    # print("texts2:", texts)
    extracted_code = [extract_code(result) for result in texts]
    request_id_list = lean4_scheduler.submit_all_request(extracted_code)
    # extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    # Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    # As soon as a task is enqueued, the next available worker process automatically picks it up.
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(random.choice(outputs_list))
    # print("output_list",outputs_list)
    # print("rewarding start")
    all_token_scores, all_token_texts, binary_pass_score,all_token_ids,tactic_mean,all_adv_masks,all_tactic_ids,all_first_error_masks = fail_aware_compute_token_level_advantages(
        prompts, completions, outputs_list,processing_class,extracted_code, parse_method,0.9,num_generation,delta0,delta1,delta2,ad_baseline=adv_baseline, score_assign=score_assign,adv_method=adv_method, first_error=first_error,potent_func=potent_func,potent_type=potent_type,potent_positive=potent_positive,shift_potential=shift_potential,potent_coef=potent_coef,entropy_position=entropy_position)
    #value_compute_token_level_advantages ,grouped_compute_token_level_advantages

    # 3. Convert to a padded tensor if desired
    #    Each row in padded_scores corresponds to one (prompt+completion) example
    #    The columns are the tokens in the completion portion
    padded_scores = rloo_list_of_lists_to_padded_tensor(all_token_scores,max_len,padding_value=0)
    all_adv_masks = rloo_list_of_lists_to_padded_tensor(all_adv_masks, max_len, padding_value=0)
    all_tactic_ids = rloo_list_of_lists_to_padded_tensor(all_tactic_ids, max_len, padding_value=-1)
    all_first_error_masks = rloo_list_of_lists_to_padded_tensor(all_first_error_masks, max_len, padding_value=0)
    binary_score = [float(item["complete"]) for item in outputs_list]
    #print("padded_scores",padded_scores.size())
    lean4_scheduler.close()

    timeout_flags = [
        1.0 if "TimeoutExpired" in (item.get("system_errors") or "") else 0.0
        for item in outputs_list
    ]





    """
    n = len(binary_score)
    num_groups = math.ceil(n / num_generation)  # works even if it doesn't divide evenly

    group_means = [
        float(np.mean(binary_score[g * num_generation: (g + 1) * num_generation]))
        for g in range(num_groups)
    ]



    for i, (scores, texts, ids,tactic_ids) in enumerate(zip(all_token_scores, all_token_texts,all_token_ids,all_tactic_ids)):
        print(f"--- Completion #{i} ---")
        if i ==0:
            baseline = group_means[0 // num_generation]
            print("baseline",baseline)
            print("output_list",outputs_list[0])
            print("completions",completions[0])
            for idx, (token_str, sc, id,tactic_ids) in enumerate(zip(texts, scores,ids,tactic_ids)):
                print(f"{idx} Token '{token_str} Token_id {id}' => Score {sc:.2f}, Tactic_ids {tactic_ids}")

    """







    return padded_scores, binary_score,all_token_ids,tactic_mean,all_adv_masks,all_tactic_ids,all_first_error_masks,timeout_flags





def lean4_outcome_reward(result):
    request_id_list = lean4_scheduler.submit_all_request(
        [re.search(r'```lean4\n(.*?)\n```', result, re.DOTALL).group(1)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)

    return outputs_list

"""
    def lean4_value_reward(result):
        request_id_list = lean4_scheduler.submit_all_request(
            [re.search(r'```lean4\n(.*?)\n```', result, re.DOTALL).group(1)])
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    
        return outputs_list
"""

def main():
    # Example data
    prompt1 =  'Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- A sequence of numbers is defined by $D_0=0,D_1=0,D_2=1$ and $D_n=D_{n-1}+D_{n-3}$ for $n\\ge 3$. What are the parities (evenness or oddness) of the triple of numbers $(D_{2021},D_{2022},D_{2023})$, where $E$ denotes even and $O$ denotes odd?\n\n$\\textbf{(A) }(O,E,O) \\qquad \\textbf{(B) }(E,E,O) \\qquad \\textbf{(C) }(E,O,E) \\qquad \\textbf{(D) }(O,O,E) \\qquad \\textbf{(E) }(O,O,O)$ Show that it is \\textbf{(C) }(E,O,E).-/\ntheorem amc12a_2021_p8 (d : ℕ → ℕ) (h₀ : d 0 = 0) (h₁ : d 1 = 0) (h₂ : d 2 = 1)\n    (h₃ : ∀ n ≥ 3, d n = d (n - 1) + d (n - 3)) : Even (d 2021) ∧ Odd (d 2022) ∧ Even (d 2023) := by\n'

    completion1 = "  /-\n  To solve the problem, we need to determine the parities of the numbers \\( D_{2021} \\), \\( D_{2022} \\), and \\( D_{2023} \\) in the sequence defined by \\( D_0 = 0 \\), \\( D_1 = 0 \\), \\( D_2 = 1 \\), and \\( D_n = D_{n-1} + D_{n-3} \\) for \\( n \\geq 3 \\).\n  1. **Initial Values**:\n     - \\( D_0 = 0 \\) (even)\n     - \\( D_1 = 0 \\) (even)\n     - \\( D_2 = 1 \\) (odd)\n  2. **Sequence Calculation**:\n     - For \\( n \\geq 3 \\), \\( D_n = D_{n-1} + D_{n-3} \\).\n  3. **Parity Patterns**:\n     - We observe the sequence's parity by calculating the first few terms:\n       - \\( D_3 = D_2 + D_0 = 1 + 0 = 1 \\) (odd)\n       - \\( D_4 = D_3 + D_1 = 1 + 0 = 1 \\) (odd)\n       - \\( D_5 = D_4 + D_2 = 1 + 1 = 2 \\) (even)\n       - \\( D_6 = D_5 + D_3 = 2 + 1 = 3 \\) (odd)\n       - \\( D_7 = D_6 + D_4 = 3 + 1 = 4 \\) (even)\n       - \\( D_8 = D_7 + D_5 = 4 + 2 = 6 \\) (even)\n       - \\( D_9 = D_8 + D_6 = 6 + 3 = 9 \\) (odd)\n  4. **Pattern Recognition**:\n     - The sequence alternates between odd and even values. Specifically, the pattern is \\( (O, E, O) \\).\n  5. **2021, 2022, 2023 Calculation**:\n     - Since the pattern \\( (O, E, O) \\) repeats every 3 terms, we can determine the parities of \\( D_{2021} \\), \\( D_{2022} \\), and \\( D_{2023} \\) by the position of these terms in the pattern.\n     - \\( 2021 \\mod 3 = 2 \\) (even)\n     - \\( 2022 \\mod 3 = 0 \\) (even)\n     - \\( 2023 \\mod 3 = 1 \\) (odd)\n  Thus, the parities are \\( (E, O, E) \\).\n  -/\n  -- Simplify the initial conditions and sequence definition\n  simp_all only [zero_add, one_add_one_eq_two]\n  -- Use the sequence definition to derive the parities of the terms\n  have := h₃ 2021 (by norm_num)\n  have := h₃ 2022 (by norm_num)\n  have := h₃ 2023 (by norm_num)\n  -- Use Aesop to solve the parity problem\n  aesop\n```"
    # Combine them into lists to match the function signatures

    prompt2='Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- Suppose $a, b, c$ are the sides of a triangle. Prove that \n\n$a^2(b+c-a)+b^2(c+a-b)+c^2(a+b-c)\\le{3abc}.$-/\ntheorem imo_1964_p2 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : c < a + b) (h₂ : b < a + c)\n    (h₃ : a < b + c) :\n    a ^ 2 * (b + c - a) + b ^ 2 * (c + a - b) + c ^ 2 * (a + b - c) ≤ 3 * a * b * c := by\n'
    completion2='  /-\n  To prove the inequality \\(a^2(b+c-a)+b^2(c+a-b)+c^2(a+b-c) \\leq 3abc\\) for the sides \\(a, b, c\\) of a triangle, we start by noting that the square of any real number is non-negative. Specifically, we consider the squares of the differences \\(a - b\\), \\(b - c\\), and \\(c - a\\). These squares are non-negative, and by summing them, we can derive the desired inequality.\n  1. The square of \\(a - b\\) is non-negative: \\((a - b)^2 \\geq 0\\).\n  2. The square of \\(b - c\\) is non-negative: \\((b - c)^2 \\geq 0\\).\n  3. The square of \\(c - a\\) is non-negative: \\((c - a)^2 \\geq 0\\).\n  By summing these inequalities and expanding the squares, we can derive the inequality \\(a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) \\leq 3abc\\). This approach leverages the properties of non-negative numbers and the structure of the triangle inequality to establish the result.\n  -/\n  -- We start by noting that the square of any real number is non-negative.\n  have h₄ := pow_two_nonneg (a - b) -- (a - b)^2 ≥ 0\n  have h₅ := pow_two_nonneg (b - c) -- (b - c)^2 ≥ 0\n  have h₆ := pow_two_nonneg (c - a) -- (c - a)^2 ≥ 0\n  -- By summing these inequalities and expanding the squares, we derive the desired inequality.\n  nlinarith\n```'

    prompt3='Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- Given $2^a = 32$ and $a^b = 125$ find $b^a$. Show that it is 243.-/\ntheorem mathd_algebra_756 (a b : ℝ) (h₀ : (2 : ℝ) ^ a = 32) (h₁ : a ^ b = 125) : b ^ a = 243 := by\n'

    completion3='  /-\n  Given \\(2^a = 32\\) and \\(a^b = 125\\), we need to find \\(b^a\\). We start by solving for \\(a\\) using the equation \\(2^a = 32\\). Taking the logarithm base 2 of both sides, we get \\(a = \\log_2 32\\). Since \\(32 = 2^5\\), we have \\(a = 5\\).\n  Next, we substitute \\(a = 5\\) into the equation \\(a^b = 125\\), yielding \\(5^b = 125\\). Taking the logarithm base 5 of both sides, we get \\(b = \\log_5 125\\). Since \\(125 = 5^3\\), we have \\(b = 3\\).\n  Finally, we need to find \\(b^a\\). Substituting \\(a = 5\\) and \\(b = 3\\), we get \\(b^a = 3^5\\). Calculating \\(3^5\\), we find \\(3^5 = 243\\).\n  -/\n  -- We start by solving for a using the equation 2^a = 32.\n  have : a = 5 := by\n    -- Taking the logarithm base 2 of both sides, we get a = log_2 32.\n    -- Since 32 = 2^5, we have a = 5.\n    apply_fun fun x : ℝ => logb 2 x at h₀\n    norm_num at h₀\n    linarith\n  -- Next, we substitute a = 5 into the equation a^b = 125.\n  subst this\n  -- Taking the logarithm base 5 of both sides, we get b = log_5 125.\n  -- Since 125 = 5^3, we have b = 3.\n  have : b = 3 := by\n    apply_fun fun x : ℝ => logb 5 x at h₁\n    norm_num at h₁\n    linarith\n  -- Finally, we need to find b^a. Substituting a = 5 and b = 3, we get b^a = 3^5.\n  subst this\n  -- Calculating 3^5, we find 3^5 = 243.\n  norm_num\n```'


    prompts = [prompt1,prompt2,prompt3]
    completions = [completion1,completion2,completion3]
    texts = [p + c for p, c in zip(prompts, completions)]
    extracted_code=[extract_code(result) for result in texts]
    print("extracted_code",extracted_code)
    # Instead of calling Lean4ServerScheduler, we define a static example of the outputs_list
    # with some errors (as you said you'd do for your test).
    outputs_list = [  {'sorries': [], 'tactics': [{'tactic': 'simp_all only [zero_add, one_add_one_eq_two]\n  -- Use the sequence definition to derive the parities of the terms', 'proofState': 0, 'pos': {'line': 40, 'column': 2}, 'goals': 'd : ℕ → ℕ\nh₀ : d 0 = 0\nh₁ : d 1 = 0\nh₂ : d 2 = 1\nh₃ : ∀ n ≥ 3, d n = d (n - 1) + d (n - 3)\n⊢ Even (d 2021) ∧ Odd (d 2022) ∧ Even (d 2023)', 'endPos': {'line': 40, 'column': 46}}], 'errors': [{'severity': 'error', 'pos': {'line': 40, 'column': 2}, 'endPos': {'line': 40, 'column': 46}, 'data': 'simp_all made no progress'}], 'warnings': [], 'infos': [], 'system_messages': '', 'system_errors': None, 'ast': {}, 'verified_code': "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- A sequence of numbers is defined by $D_0=0,D_1=0,D_2=1$ and $D_n=D_{n-1}+D_{n-3}$ for $n\\ge 3$. What are the parities (evenness or oddness) of the triple of numbers $(D_{2021},D_{2022},D_{2023})$, where $E$ denotes even and $O$ denotes odd?\n\n$\\textbf{(A) }(O,E,O) \\qquad \\textbf{(B) }(E,E,O) \\qquad \\textbf{(C) }(E,O,E) \\qquad \\textbf{(D) }(O,O,E) \\qquad \\textbf{(E) }(O,O,O)$ Show that it is \\textbf{(C) }(E,O,E).-/\ntheorem amc12a_2021_p8 (d : ℕ → ℕ) (h₀ : d 0 = 0) (h₁ : d 1 = 0) (h₂ : d 2 = 1)\n    (h₃ : ∀ n ≥ 3, d n = d (n - 1) + d (n - 3)) : Even (d 2021) ∧ Odd (d 2022) ∧ Even (d 2023) := by\n  /-\n  To solve the problem, we need to determine the parities of the numbers \\( D_{2021} \\), \\( D_{2022} \\), and \\( D_{2023} \\) in the sequence defined by \\( D_0 = 0 \\), \\( D_1 = 0 \\), \\( D_2 = 1 \\), and \\( D_n = D_{n-1} + D_{n-3} \\) for \\( n \\geq 3 \\).\n  1. **Initial Values**:\n     - \\( D_0 = 0 \\) (even)\n     - \\( D_1 = 0 \\) (even)\n     - \\( D_2 = 1 \\) (odd)\n  2. **Sequence Calculation**:\n     - For \\( n \\geq 3 \\), \\( D_n = D_{n-1} + D_{n-3} \\).\n  3. **Parity Patterns**:\n     - We observe the sequence's parity by calculating the first few terms:\n       - \\( D_3 = D_2 + D_0 = 1 + 0 = 1 \\) (odd)\n       - \\( D_4 = D_3 + D_1 = 1 + 0 = 1 \\) (odd)\n       - \\( D_5 = D_4 + D_2 = 1 + 1 = 2 \\) (even)\n       - \\( D_6 = D_5 + D_3 = 2 + 1 = 3 \\) (odd)\n       - \\( D_7 = D_6 + D_4 = 3 + 1 = 4 \\) (even)\n       - \\( D_8 = D_7 + D_5 = 4 + 2 = 6 \\) (even)\n       - \\( D_9 = D_8 + D_6 = 6 + 3 = 9 \\) (odd)\n  4. **Pattern Recognition**:\n     - The sequence alternates between odd and even values. Specifically, the pattern is \\( (O, E, O) \\).\n  5. **2021, 2022, 2023 Calculation**:\n     - Since the pattern \\( (O, E, O) \\) repeats every 3 terms, we can determine the parities of \\( D_{2021} \\), \\( D_{2022} \\), and \\( D_{2023} \\) by the position of these terms in the pattern.\n     - \\( 2021 \\mod 3 = 2 \\) (even)\n     - \\( 2022 \\mod 3 = 0 \\) (even)\n     - \\( 2023 \\mod 3 = 1 \\) (odd)\n  Thus, the parities are \\( (E, O, E) \\).\n  -/\n  -- Simplify the initial conditions and sequence definition\n  simp_all only [zero_add, one_add_one_eq_two]\n  -- Use the sequence definition to derive the parities of the terms\n  have := h₃ 2021 (by norm_num)\n  have := h₃ 2022 (by norm_num)\n  have := h₃ 2023 (by norm_num)\n  -- Use Aesop to solve the parity problem\n  aesop", 'pass': False, 'complete': False, 'verify_time': 6.8995277881622314}
,
                   

    ]

    # In a real scenario, you'd do something like:
    #
    # lean4_scheduler = Lean4ServerScheduler(...)
    # code_snippets = [re.search(r'```lean4\n(.*?)\n```', txt, re.DOTALL).group(1) for txt in texts]
    # request_id_list = lean4_scheduler.submit_all_request(code_snippets)
    # outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    #
    # But we'll skip that, as requested.


    model_name = "deepseek-ai/DeepSeek-Prover-V1.5-SFT"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Compute token-level scores
    all_token_scores, all_token_texts,binary_pass_score = compute_tactic_scores_for_output_deepseek(
        prompts, completions, outputs_list, extracted_code, tokenizer
    )


    #tactic-level scores
    all_tactic_scores, all_tactic_texts, binary=compute_token_level_advantages(
        prompts, completions, outputs_list, tokenizer, extracted_code,"advantage", 0.9
    )

    # Convert to a padded tensor if desired
    padded_token_scores = list_of_lists_to_padded_tensor(all_token_scores, padding_value=0)
   # padded_tactic_scores = list_of_lists_to_padded_tensor(all_tactic_scores, padding_value=0)
    # Print out a summary

    for i, (scores, texts) in enumerate(zip(all_token_scores, all_token_texts)):
        print(f"--- Completion #{i} ---")
        if i >= 2:
            for token_str, sc in zip(texts, scores):
                print(f"Token '{token_str}' => Score {sc:.2f}")

        # Or just look at the padded row
       # print("Padded row for this completion =>", padded_token_scores[i].tolist())



    for i, (scores, texts) in enumerate(zip(all_tactic_scores, all_tactic_texts)):
        print(f"--- Completion #{i} ---")
        if i>=2:
            for token_str, sc in zip(texts, scores):
                print(f"Tactic_Token '{token_str}' => Score {sc:.2f}")

        # Or just look at the padded row
        #print("Tactic_Padded row for this completion =>", padded_tactic_scores[i].tolist())
        #print()



    def compare_token_scores(scores1, scores2, tol=1e-6):
        """
        Compare two lists of token scores (lists of lists).
      

        Returns True if they are the same, False otherwise.
        """
        if len(scores1) != len(scores2):
            print(f"Different number of samples: {len(scores1)} vs {len(scores2)}")
            return False
        all_same = True
        for i, (s1, s2) in enumerate(zip(scores1, scores2)):
            if len(s1) != len(s2):
                print(f"Sample {i}: different number of tokens: {len(s1)} vs {len(s2)}")
                all_same = False
                continue
            for j, (score1, score2) in enumerate(zip(s1, s2)):
                if abs(score1 - score2) > tol:
                    print(f"Sample {i}, token {j}: score1 = {score1}, score2 = {score2}")
                    all_same = False
        return all_same

    # Then, after computing all_token_scores and all_tactic_scores:
    if compare_token_scores(all_token_scores, all_tactic_scores):
        print("all_token_scores and all_tactic_scores are the same.")
    else:
        print("There are differences between all_token_scores and all_tactic_scores.")

    #print("padded_scores",padded_token_scores)
    values=torch.zeros_like(padded_token_scores)
    for i in reversed(range(len(padded_token_scores[-1]))):
        next_values = values[:, i + 1] if i < len(padded_token_scores[-1]) - 1 else 0.0  # values=return in one trajectory environment
        values[:, i] = padded_token_scores[:, i] + 0.5 * next_values

    #print("values", values)
    #print("tactic_values",all_tactic_scores)
if __name__ == "__main__":
    main()