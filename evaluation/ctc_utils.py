"""
CTC Utilities for Continuous Sign Language Recognition

This module provides utilities for Connectionist Temporal Classification (CTC)
including decoding algorithms and label processing functions.

Key Components:
- Greedy CTC Decoder: Fast, deterministic decoding
- Beam Search CTC Decoder: More accurate but slower decoding
- Label processing utilities for CTC format

Usage:
    from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder
    
    # Greedy decoding
    decoded = greedy_ctc_decoder(log_probs, blank_id=105)
    
    # Beam search decoding
    decoded, score = beam_search_ctc_decoder(log_probs, blank_id=105, beam_width=10)
"""

# Standard library imports
from typing import List, Tuple, Optional
import heapq

# Third-party imports
import numpy as np
import torch


def remove_consecutive_duplicates(sequence: List[int]) -> List[int]:
    """
    Remove consecutive duplicate elements from a sequence.
    
    This is the first step in CTC decoding - collapsing repeated tokens
    that represent a single output symbol.
    
    Args:
        sequence: List of token IDs (may contain consecutive duplicates)
        
    Returns:
        List of token IDs with consecutive duplicates removed
        
    Example:
        >>> remove_consecutive_duplicates([1, 1, 2, 2, 2, 3, 1, 1])
        [1, 2, 3, 1]
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
    
    This is the second step in CTC decoding - removing the special blank
    token that was introduced for temporal alignment.
    
    Args:
        sequence: List of token IDs (may contain blank tokens)
        blank_id: ID of the blank token to remove
        
    Returns:
        List of token IDs with blank tokens removed
        
    Example:
        >>> remove_blank_tokens([1, 105, 2, 105, 105, 3], blank_id=105)
        [1, 2, 3]
    """
    return [token for token in sequence if token != blank_id]


def collapse_ctc_output(sequence: List[int], blank_id: int) -> List[int]:
    """
    Apply full CTC collapsing: remove consecutive duplicates then remove blanks.
    
    This is the standard CTC decoding operation that converts the raw CTC
    output (with blanks and repeats) into the final predicted sequence.
    
    Args:
        sequence: Raw CTC output sequence (list of token IDs)
        blank_id: ID of the blank token
        
    Returns:
        Decoded sequence without duplicates or blanks
        
    Example:
        >>> collapse_ctc_output([1, 1, 105, 2, 2, 105, 3], blank_id=105)
        [1, 2, 3]
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
    
    This is the fastest CTC decoding method. It simply takes the argmax
    at each time step and then collapses the result according to CTC rules.
    
    Process:
    1. Take argmax across vocabulary dimension at each timestep
    2. Remove consecutive duplicates
    3. Remove blank tokens
    
    Args:
        log_probs: Log probabilities from model, shape [B, T, C] or [T, B, C]
                   where B=batch, T=time, C=num_classes
        blank_id: ID of the blank token (typically num_gloss_classes)
        input_lengths: Optional tensor of actual sequence lengths [B]
                      If provided, only decode up to actual length for each sequence
        
    Returns:
        List of decoded sequences (one per batch item), where each sequence
        is a list of predicted gloss IDs
        
    Example:
        >>> log_probs = torch.randn(2, 50, 106)  # 2 sequences, 50 frames, 106 classes
        >>> decoded = greedy_ctc_decoder(log_probs, blank_id=105)
        >>> print(decoded[0])  # First decoded sequence
        [4, 17, 23, 56]
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
    
    This decoder maintains multiple candidate sequences (beams) and explores
    the most promising paths through the output space. It's more accurate than
    greedy decoding but computationally more expensive.
    
    Algorithm:
    1. Initialize beam with empty sequence
    2. For each timestep:
       a. Expand each beam with all possible next tokens
       b. Score each expanded sequence
       c. Keep only top-k sequences (beam_width)
    3. Return best sequence after CTC collapsing
    
    Args:
        log_probs: Log probabilities from model, shape [B, T, C] or [T, B, C]
        blank_id: ID of the blank token
        beam_width: Number of beams to maintain (higher = more accurate but slower)
        input_lengths: Optional tensor of actual sequence lengths [B]
        
    Returns:
        List of tuples (decoded_sequence, log_probability) for each batch item
        Each decoded_sequence is the best found sequence
        
    Note:
        This is a simplified beam search. For production, consider using
        specialized libraries like ctcdecode for better performance.
        
    Example:
        >>> log_probs = torch.randn(1, 50, 106)
        >>> decoded, score = beam_search_ctc_decoder(log_probs, blank_id=105, beam_width=10)
        >>> print(f"Decoded: {decoded[0][0]}, Score: {decoded[0][1]:.4f}")
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
    
    This is a helper function that performs beam search on one sequence.
    It maintains a priority queue of candidate sequences and explores
    the most promising paths.
    
    Args:
        log_probs: Log probabilities for single sequence [T, C]
        blank_id: ID of the blank token
        beam_width: Number of beams to maintain
        
    Returns:
        Tuple of (best_decoded_sequence, log_probability)
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
    
    Converts a list of gloss IDs into a tensor suitable for CTCLoss.
    For isolated signs, this is simply a single-element sequence.
    For continuous signs, this would be a multi-element sequence.
    
    Args:
        gloss_ids: List of gloss IDs (e.g., [42] for isolated, [3, 17, 42] for continuous)
        num_classes: Total number of gloss classes (not including blank)
        
    Returns:
        Tuple of (encoded_tensor, target_length) where:
        - encoded_tensor: 1D tensor of gloss IDs
        - target_length: Length of the sequence
        
    Example:
        >>> encode_label_sequence([42], num_classes=105)
        (tensor([42]), 1)
        >>> encode_label_sequence([3, 17, 42], num_classes=105)
        (tensor([3, 17, 42]), 3)
    """
    # Validate input
    for gloss_id in gloss_ids:
        if gloss_id < 0 or gloss_id >= num_classes:
            raise ValueError(
                f"Gloss ID {gloss_id} out of range [0, {num_classes})"
            )
    
    # Convert to tensor
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
    
    Args:
        gloss_ids: List of predicted gloss IDs
        label_mapping: Optional dictionary mapping gloss_id -> gloss_name
                      If None, returns string representation of IDs
        
    Returns:
        List of gloss names (or string IDs if mapping not provided)
        
    Example:
        >>> mapping = {42: "hello", 17: "world"}
        >>> decode_label_sequence([42, 17], mapping)
        ['hello', 'world']
        >>> decode_label_sequence([42, 17])  # Without mapping
        ['42', '17']
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
    
    WER is calculated as the Levenshtein distance (edit distance) normalized
    by the length of the reference sequence. It measures insertions, deletions,
    and substitutions needed to transform hypothesis into reference.
    
    Formula: WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    
    Args:
        reference: Ground truth sequence of gloss IDs
        hypothesis: Predicted sequence of gloss IDs
        
    Returns:
        Word Error Rate as a float (0.0 = perfect match, >1.0 = very poor)
        
    Example:
        >>> calculate_wer([1, 2, 3, 4], [1, 2, 4, 5])
        0.5  # 2 errors / 4 words
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
    
    In sign language recognition, CER is equivalent to WER when operating
    at the gloss (word) level rather than character level.
    
    Args:
        reference: Ground truth sequence of gloss IDs
        hypothesis: Predicted sequence of gloss IDs
        
    Returns:
        Character Error Rate as a float
    """
    return calculate_wer(reference, hypothesis)

