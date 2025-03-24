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
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    #print("data",data)
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
        error = item.get("error", False)  # or however you detect an "error"
        score_to_mark = -1 if error else 1
        for i in range(start_abs, end_abs):
            if 0 <= i < len(char_scores):
                char_scores[i] = score_to_mark

"""    
def compute_token_scores_for_output(prompts, completions, outputs_list, tokenizer):
        
        For each (prompt, completion, output) triple:
          - Combine prompt + completion into full_text
          - Build a per-char score array (+1 by default)
          - Mark error ranges as -1
          - Tokenize full_text (with offsets)
          - Compute token scores = average of char scores in [start, end)
          - Keep only tokens that lie within the completion portion (start >= len(prompt))
          - Return a list of token_score arrays, each item is a list of floats (or ints).
            We also return the actual text tokens if you want to see them.
    
    
        1: Characters not in errors (default).
        -1: Characters within errors (set by the messages step).
    
        
        all_token_scores = []
        all_token_texts = []  # If you want to keep track of the actual token strings
    
        for prompt, completion, out_data in zip(prompts, completions, outputs_list):
            full_text = prompt + completion
    
            # 1) Build pos_scores (per-character)
            pos_scores = [1] * len(full_text)
    
            # 3) If there are errors, mark -1
            if len(out_data["errors"]) > 0: #"errors" in out_data["errors"]:
                mark_char_scores(pos_scores, full_text,True, out_data["errors"])
    
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
    
            for tid, (start, end) in zip(input_ids, offsets):           #character-token alignment
                # If you only want tokens fully in the completion portion,
                # check if start >= prompt_len
                # (If you want partial coverage for tokens that straddle the boundary,
                # you'd have to do a partial average or something more advanced.)
                if start >= prompt_len:
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
    
            all_token_scores.append(token_scores)
            all_token_texts.append(token_texts)
            #print("all_token_scores",all_token_scores)
           # print("all_token_texts",all_token_texts)
        return all_token_scores, all_token_texts
"""


def compute_tactic_scores_for_output(prompts, completions, outputs_list, tokenizer):         #token-level reward
    """
    For each (prompt, completion, output) triple:
      - Combine prompt + completion into full_text
      - Build a per-char score array (+1 by default)
      - Mark error ranges as -1
      - Tokenize full_text (with offsets)
      - Compute token scores = average of char scores in [start, end)
      - Keep only tokens that lie within the completion portion (start >= len(prompt))
      - Return a list of token_score arrays, each item is a list of floats (or ints).
        We also return the actual text tokens if you want to see them.


    0: Characters not in tactics or errors (default).
    1: Characters within tactics (set by the tactics step).
    -1: Characters within errors (set by the messages step; overrides tactics if overlapping, since it comes later).




    """
    all_token_scores = []
    all_token_texts = []  # If you want to keep track of the actual token strings

    for i,(prompt, completion, out_data) in enumerate(zip(prompts, completions, outputs_list)):
        full_text = prompt + completion

        # 1) Build pos_scores (per-character)
        pos_scores = [0] * len(full_text)


        #2) give 1 to all tactics
        if len(out_data.get("tactics", [])) > 0:
            mark_char_scores(pos_scores, full_text, False, out_data["tactics"])

        # 3) If there are errors, mark -1
        if len(out_data.get("errors", [])) > 0:
            mark_char_scores(pos_scores, full_text, True, out_data["errors"])

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
        token_id=[]
        # if model outputs with BOS token

        for tid, (start, end) in zip(input_ids, offsets):           #character-token alignment
            # If you only want tokens fully in the completion portion,
            # check if start >= prompt_len
            # (If you want partial coverage for tokens that straddle the boundary,
            # you'd have to do a partial average or something more advanced.)
            if start >= prompt_len:                                 #if model outputs with BOS token
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
        #print("index",i)
        #print("scoring_completion_token_id", token_id)
        #print("scoring_completion_token_score", token_id)
        all_token_scores.append(token_scores)
        all_token_texts.append(token_texts)

        #print("all_token_scores",all_token_scores)
        #print("all_token_texts",all_token_texts)
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
        abs_end   = line_offsets[eL - 1] + eC
        t["abs_start"] = abs_start
        t["abs_end"]   = abs_end
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


def compute_tactic_scores_with_parents_v1(                              #tactic-level
    prompts, completions, outputs_list, tokenizer, gamma=0.8
):
    """
    Version 1:
    All parent's tokens get (parent_base_score + gamma * child_score).
    That means if a tactic has 2 children, each child's final tactic score
    is added (with discount) onto the parent.
    """
    all_token_scores = []
    all_token_texts  = []

    for prompt, completion, out_data in zip(prompts, completions, outputs_list):
        full_text = prompt + completion

        # 1) Mark the baseline +1 for tactics, -1 for errors (character-level).
        pos_scores = [0] * len(full_text)
        # 2) give 1 to all tactics
        if len(out_data.get("tactics", [])) > 0:
            mark_char_scores(pos_scores, full_text, False, out_data["tactics"])

        # 3) If there are errors, mark -1
        if len(out_data.get("errors", [])) > 0:
            mark_char_scores(pos_scores, full_text, True, out_data["errors"])

        # 2) Build the tactic tree
        tactics = out_data.get("tactics", [])
        roots   = build_tactic_tree(full_text, tactics)

        # 3) Convert pos_scores -> token_scores. We'll do that *before* we do the parent-child merges.
        encoded = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        offsets   = encoded["offset_mapping"]
        #Tokens: ['hello', 'world', '!']
        #Offsets: [(0, 5), (6, 11), (11, 12)]

        # We'll store the raw token scores initially from +1 / -1 marking.
        raw_token_scores = []
        prompt_len       = len(prompt)

        for (start, end) in offsets:
            if start >= prompt_len:
                slice_scores = pos_scores[start:end]
                avg_score    = sum(slice_scores) / len(slice_scores) if slice_scores else 0
                raw_token_scores.append(avg_score)
            else:
                # Token overlaps with prompt or is entirely in prompt => not counting for completion
                raw_token_scores.append(0.0)

        # 4) Turn these raw_token_scores into an array so we can read/write
        raw_token_scores = torch.tensor(raw_token_scores, dtype=torch.float)

        # 5) For each tactic, compute baseline tactic score = average of tokens in that region
        #    (in the completion portion only).
        #    We'll store it in tactic["_score"].
        def compute_tactic_scores_dfs(node):                                    #t
            # 1) Identify parent's “own tokens” by subtracting child ranges
            child_ranges = []
            for c in node.get("children", []):
                c_start, c_end = find_token_span_for_tactic(c, offsets, prompt_len)
                child_ranges.append((c_start, c_end))
            child_ranges.sort(key=lambda r: r[0])  # sort by start

            parent_start, parent_end = find_token_span_for_tactic(node, offsets, prompt_len)

            # subtract out children’s intervals from [parent_start, parent_end)
            parent_own_indices = []
            prev = parent_start
            for (c_start, c_end) in child_ranges:
                if c_start > prev:
                    # add [prev, c_start)
                    parent_own_indices.extend(range(prev, c_start))
                # skip child range
                prev = max(prev, c_end)
            # add leftover if there's any
            if prev < parent_end:
                parent_own_indices.extend(range(prev, parent_end))

            # gather the parent's own token scores
            parent_own_scores = [raw_token_scores[i] for i in parent_own_indices]
            if len(parent_own_scores) == 0:
                baseline_score = 0.0
            else:
                baseline_score = float(np.mean(parent_own_scores))

            # 2) Recursively score children
            child_sum = 0.0
            for c in node.get("children", []):
                child_score = compute_tactic_scores_dfs(c)
                child_sum += child_score

            final_score = baseline_score + gamma * child_sum
            node["_score"] = final_score
            return final_score

        # We define a helper to find which tokens belong to a tactic:
        def find_token_span_for_tactic(tactic, offsets, prompt_len):
            """Returns (start_idx, end_idx) in token space for the tactic's absolute [start, end)."""
            abs_start = tactic["abs_start"]
            abs_end   = tactic["abs_end"]
            # find first token whose offset range start >= abs_start
            # and last token whose offset range end <= abs_end
            # We'll do a naive linear search for brevity.
            start_idx = 0
            while start_idx < len(offsets) and offsets[start_idx][1] <= abs_start:
                start_idx += 1
            # end_idx is the first token whose offset start >= abs_end
            end_idx = start_idx
            while end_idx < len(offsets) and offsets[end_idx][0] < abs_end:
                end_idx += 1
            # Now we also need to skip tokens that lie partly in the prompt:
            # We only want tokens beyond prompt_len
            while start_idx < end_idx and offsets[start_idx][0] < prompt_len:
                start_idx += 1
            return (start_idx, end_idx)

        # 6) Compute all tactic scores from the bottom up
        for r in roots:
            compute_tactic_scores_dfs(r)

        # 7) Now that each tactic has a final score (including children),
        #    we write that final score back into the parent's tokens. In version 1,
        #    ALL parent tokens get parent_final_score.
        final_token_scores = raw_token_scores.clone()  # copy
        def apply_final_scores_dfs(node):
            final_score = node["_score"]
            (ts, te) = find_token_span_for_tactic(node, offsets, prompt_len)
            if ts < te:
                final_token_scores[ts:te] = final_score  # set them all

            for c in node.get("children", []):
                apply_final_scores_dfs(c)

        for r in roots:
            apply_final_scores_dfs(r)

        # 8) Collect the token strings and final scores
        token_texts_list = []
        final_scores_list = []
        idx_raw = 0
        for i, (start, end) in enumerate(offsets):
            if start >= prompt_len:
                token_texts_list.append(full_text[start:end])
                final_scores_list.append(final_token_scores[i].item())

        all_token_scores.append(final_scores_list)
        all_token_texts.append(token_texts_list)

    return all_token_scores, all_token_texts






def compute_tactic_scores_with_parents_bottom_up(
    prompts, completions, outputs_list, tokenizer, gamma=0.9
):
    """
    Version 1:
    All parent's tokens get (parent_base_score + gamma * child_score).
    That means if a tactic has 2 children, each child's final tactic score
    is added (with discount) onto the parent.
    """
    all_token_scores = []
    all_token_texts  = []

    for prompt, completion, out_data in zip(prompts, completions, outputs_list):
        full_text = prompt + completion

        # 1) Mark the baseline +1 for tactics, -1 for errors (character-level).
        pos_scores = [0] * len(full_text)
        if "tactics" in out_data:
            mark_char_scores(pos_scores, full_text, False, out_data["tactics"])
        if "errors" in out_data:
            mark_char_scores(pos_scores, full_text, True,  out_data["errors"])

        # 2) Build the tactic tree
        tactics = out_data.get("tactics", [])
        roots   = build_tactic_tree(full_text, tactics)

        # 3) Convert pos_scores -> token_scores. We'll do that *before* we do the parent-child merges.
        encoded = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        offsets   = encoded["offset_mapping"]
        #Tokens: ['hello', 'world', '!']
        #Offsets: [(0, 5), (6, 11), (11, 12)]

        # We'll store the raw token scores initially from +1 / -1 marking.
        raw_token_scores = []
        prompt_len       = len(prompt)

        for (start, end) in offsets:
            if start >= prompt_len:
                slice_scores = pos_scores[start:end]
                avg_score    = sum(slice_scores) / len(slice_scores) if slice_scores else 0
                raw_token_scores.append(avg_score)
            else:
                # Token overlaps with prompt or is entirely in prompt => not counting for completion
                raw_token_scores.append(0.0)

        # 4) Turn these raw_token_scores into an array so we can read/write
        raw_token_scores = torch.tensor(raw_token_scores, dtype=torch.float)

        # 5) For each tactic, compute baseline tactic score = average of tokens in that region
        #    (in the completion portion only).
        #    We'll store it in tactic["_score"].
        def compute_tactic_scores_dfs(node):
            # node is a tactic dict
            token_start, token_end = find_token_span_for_tactic(
                node, offsets, prompt_len
            )
            # average of raw_token_scores in that token range
            tactic_tokens = raw_token_scores[token_start:token_end]
            if len(tactic_tokens) == 0:
                baseline_score = 0.0
            else:
                baseline_score = tactic_tokens.mean().item()

            # compute children recursively
            child_sum = 0.0
            for c in node.get("children", []):
                child_score = compute_tactic_scores_dfs(c)
                child_sum  += child_score

            # final tactic score = baseline + gamma * sum_of_children
            final_score = baseline_score + gamma * child_sum
            node["_score"] = final_score
            return final_score

        # We define a helper to find which tokens belong to a tactic:
        def find_token_span_for_tactic(tactic, offsets, prompt_len):
            """Returns (start_idx, end_idx) in token space for the tactic's absolute [start, end)."""
            abs_start = tactic["abs_start"]
            abs_end   = tactic["abs_end"]
            # find first token whose offset range start >= abs_start
            # and last token whose offset range end <= abs_end
            # We'll do a naive linear search for brevity.
            start_idx = 0
            while start_idx < len(offsets) and offsets[start_idx][1] <= abs_start:
                start_idx += 1
            # end_idx is the first token whose offset start >= abs_end
            end_idx = start_idx
            while end_idx < len(offsets) and offsets[end_idx][0] < abs_end:
                end_idx += 1
            # Now we also need to skip tokens that lie partly in the prompt:
            # We only want tokens beyond prompt_len
            while start_idx < end_idx and offsets[start_idx][0] < prompt_len:
                start_idx += 1
            return (start_idx, end_idx)

        # 6) Compute all tactic scores from the bottom up
        for r in roots:
            compute_tactic_scores_dfs(r)

        # 7) Now that each tactic has a final score (including children),
        #    we write that final score back into the parent's tokens. In version 1,
        #    ALL parent tokens get parent_final_score.
        final_token_scores = raw_token_scores.clone()  # copy
        def apply_final_scores_dfs(node):
            final_score = node["_score"]
            (ts, te) = find_token_span_for_tactic(node, offsets, prompt_len)
            if ts < te:
                final_token_scores[ts:te] = final_score  # set them all

            for c in node.get("children", []):
                apply_final_scores_dfs(c)

        for r in roots:
            apply_final_scores_dfs(r)

        # 8) Collect the token strings and final scores
        token_texts_list = []
        final_scores_list = []
        idx_raw = 0
        for i, (start, end) in enumerate(offsets):
            if start >= prompt_len:
                token_texts_list.append(full_text[start:end])
                final_scores_list.append(final_token_scores[i].item())

        all_token_scores.append(final_scores_list)
        all_token_texts.append(token_texts_list)

    return all_token_scores, all_token_texts






def compute_tactic_scores_with_parents_v2(
    prompts, completions, outputs_list, tokenizer, gamma=0.8
):
    """
    Version 2 (generalized):
      - We compute each tactic's _baseline and _score
        (the latter includes children: baseline + gamma*sum(child) ).
      - During 'apply', we do partial coverage:
          * For each token, if it is strictly before a child's start offset,
            we add that child's final.
          * The child's own region is overwritten by the child's final.
          * Any tokens after the last child's start get no more child bonus (just parent's baseline).
    """
    all_token_scores = []
    all_token_texts = []

    for prompt, completion, out_data in zip(prompts, completions, outputs_list):
        full_text = prompt + completion

        # 1) Mark baseline (+1 tactic, -1 error)
        pos_scores = [0] * len(full_text)
        # 2) give 1 to all tactics
        if len(out_data.get("tactics", [])) > 0:
            mark_char_scores(pos_scores, full_text, False, out_data["tactics"])

        # 3) If there are errors, mark -1
        if len(out_data.get("errors", [])) > 0:
            mark_char_scores(pos_scores, full_text, True, out_data["errors"])

        # 2) Build the tactic tree
        tactics = out_data.get("tactics", [])
        roots   = build_tactic_tree(full_text, tactics)

        # 3) Convert pos_scores -> raw token-level array
        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        offsets = encoded["offset_mapping"]
        prompt_len = len(prompt)

        raw_token_scores_list = []
        for (start, end) in offsets:
            if start >= prompt_len:
                slice_scores = pos_scores[start:end]
                if slice_scores:
                    avg_score = sum(slice_scores) / len(slice_scores)
                else:
                    avg_score = 0.0
                raw_token_scores_list.append(avg_score)
            else:
                raw_token_scores_list.append(0.0)
        raw_token_scores = torch.tensor(raw_token_scores_list, dtype=torch.float)

        # 4) DFS to compute each node's baseline and final
        def find_token_span_for_tactic(tactic):
            abs_start = tactic["abs_start"]
            abs_end   = tactic["abs_end"]
            start_idx = 0
            while start_idx < len(offsets) and offsets[start_idx][1] <= abs_start:
                start_idx += 1
            end_idx = start_idx
            while end_idx < len(offsets) and offsets[end_idx][0] < abs_end:
                end_idx += 1
            # skip tokens in prompt
            while start_idx < end_idx and offsets[start_idx][0] < prompt_len:
                start_idx += 1
            return (start_idx, end_idx)

        def compute_tactic_scores_dfs(node):
            ts, te = find_token_span_for_tactic(node)
            if te > ts:
                baseline = raw_token_scores[ts:te].mean().item()
            else:
                baseline = 0.0
            node["_baseline"] = baseline

            child_sum = 0.0
            for c in node.get("children", []):
                child_sum += compute_tactic_scores_dfs(c)

            final_score = baseline + gamma * child_sum
            node["_score"] = final_score
            return final_score

        for r in roots:
            compute_tactic_scores_dfs(r)

        # 5) Now apply partial coverage
        final_token_scores = raw_token_scores.clone()

        def apply_partial_coverage(node):
            """
            We'll treat the parent's region as segments separated by child start offsets:
              - segment from parent's start to child1 start => baseline + sum(of all children that haven't started yet)
              - child's region => child's final (recursively)
              - segment from child1 end to child2 start => baseline + sum(of children that haven't started yet), etc.
              - after last child => baseline (no more children upcoming).
            """
            node_ts, node_te = find_token_span_for_tactic(node)
            parent_baseline  = node["_baseline"]
            children_info = []
            for c in node.get("children", []):
                c_ts, c_te = find_token_span_for_tactic(c)
                children_info.append((c, c_ts, c_te))
            # sort children by start offset
            children_info.sort(key=lambda x: x[1])

            pointer = node_ts

            # We'll keep a "remaining children" set in ascending start order
            # so we can see which children haven't "started" yet.
            # Because once we cross child i's start, we no longer add child i's final to subsequent tokens.
            remaining = children_info[:]  # copy

            for idx, (child, c_ts, c_te) in enumerate(children_info):
                if pointer < c_ts and pointer < node_te:
                    segment_end = min(c_ts, node_te)
                    # among 'remaining', find which children have c_j_ts >= pointer
                    # i.e. haven't started yet
                    sum_of_child_final = 0.0
                    for (c_j, c_j_ts, c_j_te) in remaining:
                        # If c_j_ts > pointer, we are "before" that child
                        # For equality edge cases, decide whether "before" means strictly < or ≤
                        # We'll do strictly <. So if c_j_ts == pointer, that means we are exactly at child start => not before.
                        if c_j_ts > pointer:
                            sum_of_child_final += c_j["_score"]

                    # parent's baseline + sum_of_child_final
                    final_token_scores[pointer:segment_end] = parent_baseline + sum_of_child_final      #??gamma????

                # now we apply the child's region
                apply_partial_coverage(child)

                # move pointer to child's end
                pointer = max(pointer, c_te)
                # we've "passed" child start, so remove it from 'remaining'
                if child in [rc[0] for rc in remaining]:
                    remaining.remove((child, c_ts, c_te))

            # after the last child, we do [pointer, node_te)
            if pointer < node_te:
                sum_of_child_final = 0.0
                for (c_j, c_j_ts, c_j_te) in remaining:
                    if c_j_ts > pointer:
                        sum_of_child_final += c_j["_score"]
                final_token_scores[pointer:node_te] = parent_baseline + sum_of_child_final


        for root in roots:
            apply_partial_coverage(root)


        # 6) Collect final tokens in the completion portion
        token_texts_list  = []
        final_scores_list = []
        for i, (start, end) in enumerate(offsets):
            if start >= prompt_len:
                token_texts_list.append(full_text[start:end])
                final_scores_list.append(final_token_scores[i].item())

        all_token_scores.append(final_scores_list)
        all_token_texts.append(token_texts_list)

    return all_token_scores, all_token_texts




def list_of_lists_to_padded_tensor(list_of_lists, padding_value=0):
    """
    Convert list of variable-length lists to a padded 2D Tensor
    shape: (batch_size, max_length_in_batch)
    """
    for i,seq in enumerate(list_of_lists):
        print("index",i)
        print("len(seq)",len(seq))
    max_len = max(len(seq) for seq in list_of_lists) if list_of_lists else 0
    batch_size = len(list_of_lists)
    padded_tensor = torch.full((batch_size, max_len), fill_value=padding_value, dtype=torch.float)

    for i, seq in enumerate(list_of_lists):
        length = len(seq)
        padded_tensor[i, :length] = torch.tensor(seq, dtype=torch.float)

    return padded_tensor
def extract_code(inputs):
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except:
        return "None"

def lean4_value_reward(prompts, completions, processing_class):
    texts = [p + c for p, c in zip(prompts, completions)]
    #print("texts1:",texts)
    #print("type",type(texts[0]))
    #print("\n\n")
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=8, timeout=300, memory_limit=10, name='verifier')
    #print("texts2:", texts)
    request_id_list = lean4_scheduler.submit_all_request([extract_code(result) for result in texts])
    #extract lean code in the output and give to lean4_scheduler.submit_all_request, after this, each input goes to queue, and request_id_list receive each id.
    #Worker processes (Lean4ServerProcess) are already running Since p.start() was called in Lean4ServerScheduler.__init__(), all workers are already in their run() loops.
    #As soon as a task is enqueued, the next available worker process automatically picks it up.

    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print("rewarding start")
    all_token_scores, all_token_texts = compute_tactic_scores_for_output(
        prompts, completions, outputs_list, processing_class)

    # 3. Convert to a padded tensor if desired
    #    Each row in padded_scores corresponds to one (prompt+completion) example
    #    The columns are the tokens in the completion portion
    padded_scores = list_of_lists_to_padded_tensor(all_token_scores, padding_value=0)
    print("padded_scores",padded_scores.size())
    lean4_scheduler.close()
    return padded_scores



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
    prompt = "example (m n : Nat) : m - n = 0 ∨ m ≠ n := by\n"

    completion = """  cases Decidable.em (m = n) with --m = n ∨ ¬m = n
    |inl heq => rw [heq]; apply Or.inl; exact Nat.sub_self
    |inr hne => apply Or.inr;"""

    # Combine them into lists to match the function signatures
    prompts = [prompt,prompt]
    completions = [completion,completion]

    # Instead of calling Lean4ServerScheduler, we define a static example of the outputs_list
    # with some errors (as you said you'd do for your test).
    outputs_list = [
        {"tactics":
             [{"usedConstants": ["Decidable.em", "Nat", "instDecidableEqNat", "Eq"],
               "tactic":
                   "cases Decidable.em (m = n) with\n  --m = n ∨ ¬m = n\n| inl heq => rw [heq]; apply Or.inl; exact Nat.sub_self\n| inr hne => apply Or.inr;\n  /-example (m n : Nat) : m - n = 0 ∨ m ≠ n := by\n    cases Decidable.em (m=n) with --m = n ∨ ¬m = n\n      |inr he => rw [heq]; apply Or.inl; exact Nat.sub_self\n      |inr hne => apply Or.inr;\n  -/",
               "proofState": 0,
               "pos": {"line": 2, "column": 2},
               "goals": "m n : Nat\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 29}},
              {"usedConstants": [],
               "tactic":
                   "cases Decidable.em (m = n) with\n  --m = n ∨ ¬m = n\n| inl heq => rw [heq]; apply Or.inl; exact Nat.sub_self\n| inr hne => apply Or.inr;\n  /-example (m n : Nat) : m - n = 0 ∨ m ≠ n := by\n    cases Decidable.em (m=n) with --m = n ∨ ¬m = n\n      |inr he => rw [heq]; apply Or.inl; exact Nat.sub_self\n      |inr hne => apply Or.inr;\n  -/",
               "proofState": 1,
               "pos": {"line": 2, "column": 2},
               "goals": "m n : Nat\nx✝ : m = n ∨ ¬m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 29}},
              {"usedConstants":
                   ["Eq.mpr",
                    "congrArg",
                    "HSub.hSub",
                    "id",
                    "instSubNat",
                    "Ne",
                    "instOfNatNat",
                    "instHSub",
                    "Nat",
                    "Or",
                    "OfNat.ofNat",
                    "Eq"],
               "tactic": "rw [heq]",
               "proofState": 2,
               "pos": {"line": 3, "column": 16},
               "goals": "case inl\nm n : Nat\nheq : m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 3, "column": 24}},
              {"usedConstants": ["Or.inl"],
               "tactic": "apply Or.inl",
               "proofState": 3,
               "pos": {"line": 3, "column": 26},
               "goals": "case inl\nm n : Nat\nheq : m = n\n⊢ n - n = 0 ∨ n ≠ n",
               "endPos": {"line": 3, "column": 38}},
              {"usedConstants": [],
               "tactic": "exact Nat.sub_self",
               "proofState": 4,
               "pos": {"line": 3, "column": 40},
               "goals": "case inl.h\nm n : Nat\nheq : m = n\n⊢ n - n = 0",
               "endPos": {"line": 3, "column": 58}},
              {"usedConstants": ["Or.inr"],
               "tactic": "apply Or.inr",
               "proofState": 5,
               "pos": {"line": 4, "column": 16},
               "goals": "case inr\nm n : Nat\nhne : ¬m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 28}}],
         "messages":
             [{"severity": "error",
               "pos": {"line": 3, "column": 40},
               "endPos": {"line": 3, "column": 58},
               "data":
                   "type mismatch\n  Nat.sub_self\nhas type\n  ∀ (n : Nat), n - n = 0 : Prop\nbut is expected to have type\n  n - n = 0 : Prop"},
              {"severity": "error",
               "pos": {"line": 4, "column": 13},
               "endPos": {"line": 4, "column": 29},
               "data": "unsolved goals\ncase inr.h\nm n : Nat\nhne : ¬m = n\n⊢ m ≠ n"}],
         "env": 0}, {"tactics":
             [{"usedConstants": ["Decidable.em", "Nat", "instDecidableEqNat", "Eq"],
               "tactic":
                   "cases Decidable.em (m = n) with\n  --m = n ∨ ¬m = n\n| inl heq => rw [heq]; apply Or.inl; exact Nat.sub_self\n| inr hne => apply Or.inr;\n  /-example (m n : Nat) : m - n = 0 ∨ m ≠ n := by\n    cases Decidable.em (m=n) with --m = n ∨ ¬m = n\n      |inr he => rw [heq]; apply Or.inl; exact Nat.sub_self\n      |inr hne => apply Or.inr;\n  -/",
               "proofState": 0,
               "pos": {"line": 2, "column": 2},
               "goals": "m n : Nat\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 29}},
              {"usedConstants": [],
               "tactic":
                   "cases Decidable.em (m = n) with\n  --m = n ∨ ¬m = n\n| inl heq => rw [heq]; apply Or.inl; exact Nat.sub_self\n| inr hne => apply Or.inr;\n  /-example (m n : Nat) : m - n = 0 ∨ m ≠ n := by\n    cases Decidable.em (m=n) with --m = n ∨ ¬m = n\n      |inr he => rw [heq]; apply Or.inl; exact Nat.sub_self\n      |inr hne => apply Or.inr;\n  -/",
               "proofState": 1,
               "pos": {"line": 2, "column": 2},
               "goals": "m n : Nat\nx✝ : m = n ∨ ¬m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 29}},
              {"usedConstants":
                   ["Eq.mpr",
                    "congrArg",
                    "HSub.hSub",
                    "id",
                    "instSubNat",
                    "Ne",
                    "instOfNatNat",
                    "instHSub",
                    "Nat",
                    "Or",
                    "OfNat.ofNat",
                    "Eq"],
               "tactic": "rw [heq]",
               "proofState": 2,
               "pos": {"line": 3, "column": 16},
               "goals": "case inl\nm n : Nat\nheq : m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 3, "column": 24}},
              {"usedConstants": ["Or.inl"],
               "tactic": "apply Or.inl",
               "proofState": 3,
               "pos": {"line": 3, "column": 26},
               "goals": "case inl\nm n : Nat\nheq : m = n\n⊢ n - n = 0 ∨ n ≠ n",
               "endPos": {"line": 3, "column": 38}},
              {"usedConstants": [],
               "tactic": "exact Nat.sub_self",
               "proofState": 4,
               "pos": {"line": 3, "column": 40},
               "goals": "case inl.h\nm n : Nat\nheq : m = n\n⊢ n - n = 0",
               "endPos": {"line": 3, "column": 58}},
              {"usedConstants": ["Or.inr"],
               "tactic": "apply Or.inr",
               "proofState": 5,
               "pos": {"line": 4, "column": 16},
               "goals": "case inr\nm n : Nat\nhne : ¬m = n\n⊢ m - n = 0 ∨ m ≠ n",
               "endPos": {"line": 4, "column": 28}}],
         "messages":
             [{"severity": "error",
               "pos": {"line": 3, "column": 40},
               "endPos": {"line": 3, "column": 58},
               "data":
                   "type mismatch\n  Nat.sub_self\nhas type\n  ∀ (n : Nat), n - n = 0 : Prop\nbut is expected to have type\n  n - n = 0 : Prop"},
              {"severity": "error",
               "pos": {"line": 4, "column": 13},
               "endPos": {"line": 4, "column": 29},
               "data": "unsolved goals\ncase inr.h\nm n : Nat\nhne : ¬m = n\n⊢ m ≠ n"}],
         "env": 0}
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
    all_token_scores, all_token_texts = compute_tactic_scores_for_output(
        prompts, completions, outputs_list, tokenizer
    )


    #tactic-level scores
    all_tactic_scores, all_tactic_texts=compute_tactic_scores_with_parents_v1(
        prompts, completions, outputs_list, tokenizer
    )

    # Convert to a padded tensor if desired
    padded_token_scores = list_of_lists_to_padded_tensor(all_token_scores, padding_value=0)
    padded_tactic_scores = list_of_lists_to_padded_tensor(all_tactic_scores, padding_value=0)
    # Print out a summary
    for i, (scores, texts) in enumerate(zip(all_token_scores, all_token_texts)):
        #print(f"--- Completion #{i} ---")
        for token_str, sc in zip(texts, scores):
            print(f"Token '{token_str}' => Score {sc:.2f}")

        # Or just look at the padded row
        #print("Padded row for this completion =>", padded_token_scores[i].tolist())
       # print()


    for i, (scores, texts) in enumerate(zip(all_tactic_scores, all_tactic_texts)):
        #print(f"--- Completion #{i} ---")
        for token_str, sc in zip(texts, scores):
            print(f"Tactic_Token '{token_str}' => Score {sc:.2f}")

        # Or just look at the padded row
        #print("Tactic_Padded row for this completion =>", padded_tactic_scores[i].tolist())
        print()


    #print("padded_scores",padded_token_scores)
    values=torch.zeros_like(padded_token_scores)
    for i in reversed(range(len(padded_token_scores[-1]))):
        next_values = values[:, i + 1] if i < len(padded_token_scores[-1]) - 1 else 0.0  # values=return in one trajectory environment
        values[:, i] = padded_token_scores[:, i] + 0.5 * next_values

    #print("values", values)
    #print("tactic_values",all_tactic_scores)
if __name__ == "__main__":
    main()