"""
CTC utilities for continuous sign language recognition.

Provides decoding algorithms and label processing functions for CTC models.
"""

# Standard library imports
from typing import List, Tuple, Optional, Dict
import heapq

# Third-party imports
import numpy as np
import torch


def remove_consecutive_duplicates(sequence: List[int]) -> List[int]:
    """
    Remove consecutive duplicate elements from a sequence.
    
    First step in CTC decoding - collapsing repeated tokens.
    """
    if len(sequence) == 0:
        return []
    
    collapsed = [sequence[0]]
    for token in sequence[1:]:
        if token != collapsed[-1]:
            collapsed.append(token)
    
    return collapsed


def remove_blank_tokens(sequence: List[int], blank_id: int) -> List[int]:
    """
    Remove blank tokens from a sequence.
    
    Second step in CTC decoding - removing blank tokens used for temporal alignment.
    """
    return [token for token in sequence if token != blank_id]


def collapse_ctc_output(sequence: List[int], blank_id: int) -> List[int]:
    """
    Apply full CTC collapsing: remove consecutive duplicates then remove blanks.
    
    Converts raw CTC output into final predicted sequence.
    """
    # Step 1: Remove consecutive duplicates
    collapsed = remove_consecutive_duplicates(sequence)
    
    # Step 2: Remove blank tokens
    decoded = remove_blank_tokens(collapsed, blank_id)
    
    return decoded


def greedy_ctc_decoder(
    log_probs: torch.Tensor,
    blank_id: int,
    input_lengths: Optional[torch.Tensor] = None
) -> List[List[int]]:
    """
    Greedy CTC decoder - selects most likely token at each timestep.
    
    Takes argmax at each timestep, removes consecutive duplicates and blanks.
    """
    # Ensure log_probs is in [B, T, C] format
    if log_probs.dim() == 3:
        if log_probs.size(1) < log_probs.size(0):
            # Likely [T, B, C], transpose to [B, T, C]
            log_probs = log_probs.permute(1, 0, 2)
    
    # Get dimensions
    batch_size = log_probs.size(0)
    seq_length = log_probs.size(1)
    
    # Step 1: Get the most likely token at each timestep
    # Shape: [B, T]
    best_path = torch.argmax(log_probs, dim=2)
    
    # Convert to numpy for easier manipulation
    best_path = best_path.cpu().numpy()
    
    # Step 2: Decode each sequence in the batch
    decoded_sequences = []
    
    for batch_idx in range(batch_size):
        # Get the sequence for this batch item
        sequence = best_path[batch_idx]
        
        # Use only actual length if provided
        if input_lengths is not None:
            actual_length = input_lengths[batch_idx].item()
            sequence = sequence[:actual_length]
        
        # Apply CTC collapsing (remove duplicates and blanks)
        decoded = collapse_ctc_output(sequence.tolist(), blank_id)
        
        decoded_sequences.append(decoded)
    
    return decoded_sequences


def beam_search_ctc_decoder(
    log_probs: torch.Tensor,
    blank_id: int,
    beam_width: int = 10,
    input_lengths: Optional[torch.Tensor] = None
) -> List[Tuple[List[int], float]]:
    """
    Beam search CTC decoder - explores multiple hypotheses for better accuracy.
    
    Maintains multiple candidate sequences and explores the most promising paths.
    More accurate than greedy decoding but computationally more expensive.
    """
    # Ensure log_probs is in [B, T, C] format
    if log_probs.dim() == 3:
        if log_probs.size(1) < log_probs.size(0):
            log_probs = log_probs.permute(1, 0, 2)
    
    batch_size = log_probs.size(0)
    
    # Decode each sequence in the batch
    results = []
    
    for batch_idx in range(batch_size):
        # Get log probs for this sequence [T, C]
        seq_log_probs = log_probs[batch_idx]
        
        # Use only actual length if provided
        if input_lengths is not None:
            actual_length = input_lengths[batch_idx].item()
            seq_log_probs = seq_log_probs[:actual_length]
        
        # Decode this sequence
        decoded_seq, score = _beam_search_single(
            seq_log_probs, blank_id, beam_width
        )
        
        results.append((decoded_seq, score))
    
    return results


def _beam_search_single(
    log_probs: torch.Tensor,
    blank_id: int,
    beam_width: int
) -> Tuple[List[int], float]:
    """
    Beam search decoder for a single sequence.
    
    Maintains a priority queue of candidate sequences and explores the most promising paths.
    """
    T, C = log_probs.shape
    
    # Convert to numpy for easier manipulation
    log_probs_np = log_probs.cpu().numpy()
    
    # Initialize beams: (score, timestep, path)
    # Each beam is: (cumulative_log_prob, path_as_list)
    beams = [(0.0, [])]  # Start with empty path
    
    # Process each timestep
    for t in range(T):
        new_beams = []
        
        # Expand each current beam
        for score, path in beams:
            # Consider all possible next tokens
            for c in range(C):
                new_score = score + log_probs_np[t, c]
                new_path = path + [c]
                new_beams.append((new_score, new_path))
        
        # Keep only top beam_width sequences
        # Sort by score (descending) and take top-k
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
    
    # Get the best beam
    best_score, best_path = beams[0]
    
    # Apply CTC collapsing to get final sequence
    decoded = collapse_ctc_output(best_path, blank_id)
    
    return decoded, best_score


def encode_label_sequence(
    gloss_ids: List[int],
    num_classes: int
) -> Tuple[torch.Tensor, int]:
    """
    Encode a sequence of gloss IDs for CTC training.
    
    Target sequences contain only gloss IDs (0 to num_classes-1).
    The blank token is handled internally by CTCLoss.
    """
    # Validate input
    for gloss_id in gloss_ids:
        if gloss_id < 0 or gloss_id >= num_classes:
            raise ValueError(f"Gloss ID {gloss_id} out of range [0, {num_classes})")
    
    encoded = torch.tensor(gloss_ids, dtype=torch.long)
    target_length = len(gloss_ids)
    
    return encoded, target_length


def decode_label_sequence(
    gloss_ids: List[int],
    label_mapping: Optional[dict] = None
) -> List[str]:
    """
    Decode a sequence of gloss IDs to human-readable labels.
    
    Converts predicted gloss IDs back to their text representations.
    """
    if label_mapping is None:
        # Return string representation of IDs
        return [str(gid) for gid in gloss_ids]
    
    # Map IDs to names
    decoded = []
    for gid in gloss_ids:
        if gid in label_mapping:
            decoded.append(label_mapping[gid])
        else:
            # Fallback to ID if not in mapping
            decoded.append(f"<unk_{gid}>")
    
    return decoded


def calculate_wer(reference: List[int], hypothesis: List[int]) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis sequences.
    
    WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=reference length.
    """
    # Handle edge cases
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else float('inf')
    
    # Calculate Levenshtein distance using dynamic programming
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    # Initialize DP table
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    # Base cases
    for i in range(ref_len + 1):
        dp[i][0] = i  # Deletions
    for j in range(hyp_len + 1):
        dp[0][j] = j  # Insertions
    
    # Fill DP table
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                # Match - no cost
                dp[i][j] = dp[i-1][j-1]
            else:
                # Minimum of substitution, deletion, insertion
                dp[i][j] = 1 + min(
                    dp[i-1][j-1],  # Substitution
                    dp[i-1][j],    # Deletion
                    dp[i][j-1]     # Insertion
                )
    
    # Calculate WER
    edit_distance = dp[ref_len][hyp_len]
    wer = edit_distance / ref_len
    
    return wer


def calculate_cer(reference: List[int], hypothesis: List[int]) -> float:
    """
    Calculate Character Error Rate (CER) - alias for WER at gloss level.
    
    In sign language recognition, CER is equivalent to WER at the gloss level.
    """
    return calculate_wer(reference, hypothesis)


def calculate_wer_and_errors(reference: List[int], hypothesis: List[int]) -> Tuple[float, Dict[str, int]]:
    """
    Calculate WER with detailed error breakdown.
    
    Returns tuple of (wer, errors_dict) where errors_dict contains
    'S' (substitutions), 'D' (deletions), 'I' (insertions).
    """
    if len(reference) == 0:
        wer = 0.0 if len(hypothesis) == 0 else float('inf')
        return wer, {'S': 0, 'D': 0, 'I': len(hypothesis)}
    
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    ops = [[None] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    for i in range(ref_len + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(hyp_len + 1):
        dp[0][j] = j
        ops[0][j] = 'I'
    ops[0][0] = None
    
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'M'
            else:
                sub_cost = dp[i-1][j-1] + 1
                del_cost = dp[i-1][j] + 1
                ins_cost = dp[i][j-1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                
                if min_cost == sub_cost:
                    ops[i][j] = 'S'
                elif min_cost == del_cost:
                    ops[i][j] = 'D'
                else:
                    ops[i][j] = 'I'
    
    i, j = ref_len, hyp_len
    insertions = deletions = substitutions = 0
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'M':
            i -= 1
            j -= 1
        elif op == 'S':
            substitutions += 1
            i -= 1
            j -= 1
        elif op == 'D':
            deletions += 1
            i -= 1
        elif op == 'I':
            insertions += 1
            j -= 1
        else:
            break
    
    wer = dp[ref_len][hyp_len] / ref_len
    return wer, {'S': substitutions, 'D': deletions, 'I': insertions}

