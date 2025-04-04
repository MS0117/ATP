



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
