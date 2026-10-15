import math
from typing import Dict, Union

def kl_divergence_from_logprobs(
    p_logprobs: Dict[str, float],
    q_logprobs: Dict[str, float],
    alpha: float = 1e-10,
    min_logprob: float = -30.0
) -> float:
    """
    Calculate KL divergence KL(P||Q) from two dictionaries of log probabilities.
    
    Args:
        p_logprobs: Dictionary mapping keys to log probabilities for distribution P
        q_logprobs: Dictionary mapping keys to log probabilities for distribution Q
        alpha: Smoothing parameter (probability mass) for missing keys
        min_logprob: Minimum log probability to use for missing keys (alternative to alpha)
    
    Returns:
        KL divergence KL(P||Q)
    
    Note:
        - Uses alpha smoothing: assigns probability 'alpha' to missing keys
        - If a key is missing from Q but present in P, uses min_logprob or log(alpha)
        - The distributions are renormalized after smoothing
    """
    
    # Get all unique keys from both distributions
    all_keys = set(p_logprobs.keys()) | set(q_logprobs.keys())
    
    # Use the more negative value between min_logprob and log(alpha)
    smooth_logprob = min(min_logprob, math.log(alpha))
    
    # Build complete distributions with smoothing for missing keys
    p_complete = {}
    q_complete = {}
    
    for key in all_keys:
        p_complete[key] = p_logprobs.get(key, smooth_logprob)
        q_complete[key] = q_logprobs.get(key, smooth_logprob)
    
    # Convert log probabilities to probabilities and normalize
    p_probs = {}
    q_probs = {}
    
    # Convert to probabilities
    p_sum = sum(math.exp(logp) for logp in p_complete.values())
    q_sum = sum(math.exp(logp) for logp in q_complete.values())
    
    # Normalize
    for key in all_keys:
        p_probs[key] = math.exp(p_complete[key]) / p_sum
        q_probs[key] = math.exp(q_complete[key]) / q_sum
    
    # Calculate KL divergence
    kl_div = 0.0
    for key in all_keys:
        if p_probs[key] > 0:  # Avoid log(0)
            # KL(P||Q) = sum_x p(x) * log(p(x)/q(x))
            kl_div += p_probs[key] * math.log(p_probs[key] / q_probs[key])
    
    return kl_div


def kl_divergence_from_logprobs_simple(
    p_logprobs: Dict[str, float],
    q_logprobs: Dict[str, float],
    missing_logprob: float = -20.0
) -> float:
    """
    Simpler version that doesn't renormalize - assumes input distributions are already normalized.
    
    Args:
        p_logprobs: Dictionary mapping keys to log probabilities for distribution P
        q_logprobs: Dictionary mapping keys to log probabilities for distribution Q
        missing_logprob: Log probability to assign to missing keys
    
    Returns:
        KL divergence KL(P||Q)
    """
    
    kl_div = 0.0
    
    # Only iterate over keys in P (since KL(P||Q) is expectation under P)
    for key, p_logprob in p_logprobs.items():
        q_logprob = q_logprobs.get(key, missing_logprob)
        
        # Convert to probabilities
        p_prob = math.exp(p_logprob)
        
        # KL contribution: p(x) * (log p(x) - log q(x))
        kl_div += p_prob * (p_logprob - q_logprob)
    
    return kl_div