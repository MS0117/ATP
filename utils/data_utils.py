import torch
from typing import Dict, List
from transformers import AutoTokenizer


DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "bfloat16":torch.bfloat16, "float16":torch.float16, "float32":torch.float32}


def make_padded_logits(
        attention_mask: torch.LongTensor,
        logits_to_pad: List[torch.FloatTensor]
) -> torch.FloatTensor:
    attention_mask_ = attention_mask[:, :-1]
    batch_size, seq_len = attention_mask_.shape

    # Create a new attention mask with the same size as the original
    new_attention_mask = torch.zeros_like(attention_mask_, dtype=torch.float64)

    for i in range(batch_size):
        # Find where the original mask becomes 0
        nonzero_count = attention_mask_[i].nonzero().shape[0]

        # Calculate how many floats we can append within the sequence length
        append_len = min(seq_len - nonzero_count, logits_to_pad[i].shape[0])

        # Append the floats
        new_attention_mask[i, nonzero_count:nonzero_count + append_len] = logits_to_pad[i][:append_len]

    return new_attention_mask
