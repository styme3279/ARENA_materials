"""
The code is structured such that for all of the functions (or model running)
that takes more than a minute, the function outputs will be automatically cached.
You can run this script to prepare that cache for everything involving attention weights.
Then run the other scripts analyzing the data or plotting the results.

Usage:
    python prep_attn_cache.py                    # Cache for default models
    python prep_attn_cache.py --model qwen-15b   # Cache for specific model
    python prep_attn_cache.py --all-models       # Cache for all available models
    python prep_attn_cache.py --kl-divergence    # Also cache KL divergences
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import (
    get_all_heads_vert_scores,
    get_all_receiver_head_scores,
    get_problem_text_sentences,
    get_model_rollouts_root,
)
from attention_analysis.attn_funcs import get_avg_attention_matrix
from attention_analysis.attn_supp_funcs import get_suppression_KL_matrix
from pytorch_models.model_config import model2layers_heads


def get_all_problems(model_name: str) -> List[Tuple[str, bool]]:
    problems = []
    dir_root = get_model_rollouts_root(model_name)
    
    correct_dir = os.path.join(dir_root, "correct_base_solution")
    if os.path.exists(correct_dir):
        for problem_dir in os.listdir(correct_dir):
            if problem_dir.startswith("problem_"):
                problems.append((problem_dir, True))
    
    incorrect_dir = os.path.join(dir_root, "incorrect_base_solution")
    if os.path.exists(incorrect_dir):
        for problem_dir in os.listdir(incorrect_dir):
            if problem_dir.startswith("problem_"):
                problems.append((problem_dir, False))
    
    return sorted(problems)


def cache_attention_matrices(
    model_name: str,
    problems: List[Tuple[str, bool]],
    verbose: bool = True
) -> None:
    """Cache attention matrices for all problems and all layer-head combinations."""
    layers, heads = model2layers_heads(model_name)
    
    print(f"\nCaching attention matrices for {model_name}...")
    print(f"  Model has {layers} layers and {heads} heads")
    print(f"  Processing {len(problems)} problems")
    
    for problem_num, is_correct in tqdm(problems, desc="Problems"):
        try:
            text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
            
            if verbose:
                status = "correct" if is_correct else "incorrect"
                print(f"\n  Processing {problem_num} ({status}): {len(sentences)} sentences")
            
            for layer in range(layers):
                for head in range(heads):
                    _ = get_avg_attention_matrix(
                        text,
                        model_name=model_name,
                        layer=layer,
                        head=head,
                        sentences=sentences,
                    )
            
        except Exception as e:
            print(f"  Error processing {problem_num}: {e}")
            continue


def cache_vertical_scores(
    model_name: str,
    problems: List[Tuple[str, bool]],
    proximity_ignore: int = 4,
    control_depth: bool = False,
    verbose: bool = True
) -> None:
    """Cache vertical attention scores for all problems."""
    print(f"\nCaching vertical scores for {model_name}...")
    print(f"  proximity_ignore={proximity_ignore}, control_depth={control_depth}")
    
    for problem_num, is_correct in tqdm(problems, desc="Problems"):
        try:
            text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
            
            if verbose:
                status = "correct" if is_correct else "incorrect"
                print(f"\n  Processing {problem_num} ({status}): {len(sentences)} sentences")
            
            _ = get_all_heads_vert_scores(
                text,
                sentences,
                model_name=model_name,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
                score_type="mean",
            )
            
        except Exception as e:
            print(f"  Error processing {problem_num}: {e}")
            continue


def cache_receiver_head_scores(
    model_name: str,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    top_k: int = 20,
) -> None:
    """Cache receiver head scores for all problems."""
    print(f"\nCaching receiver head scores for {model_name}...")
    print(f"  top_k={top_k}, proximity_ignore={proximity_ignore}, control_depth={control_depth}")
    
    _ = get_all_receiver_head_scores(
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
        top_k=top_k,
    )


def cache_kl_divergences(
    model_name: str,
    problems: List[Tuple[str, bool]],
    p_nucleus: float = 0.9999,
    take_log: bool = True,
    verbose: bool = True
) -> None:
    """Cache KL divergences between sentences for all problems."""
    print(f"\nCaching KL divergences for {model_name}...")
    print(f"  p_nucleus={p_nucleus}, take_log={take_log}")
    
    for problem_num, is_correct in tqdm(problems, desc="Problems"):
        try:
            if problem_num.startswith("problem_"):
                problem_num_int = int(problem_num.replace("problem_", ""))
            else:
                problem_num_int = int(problem_num)
            
            if verbose:
                status = "correct" if is_correct else "incorrect"
                print(f"\n  Processing problem {problem_num_int} ({status})")
            
            # This will automatically cache through the @pkld decorator
            _ = get_suppression_KL_matrix(
                problem_num=problem_num_int,
                p_nucleus=p_nucleus,
                model_name=model_name,
                is_correct=is_correct,
                only_first=None,
                take_log=take_log,
            )
            
        except Exception as e:
            print(f"  Error processing problem {problem_num_int}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache attention weights and related computations for analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to cache (e.g., qwen-15b, llama-8b)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Cache for all available models",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip caching raw attention matrices",
    )
    parser.add_argument(
        "--skip-vertical",
        action="store_true",
        help="Skip caching vertical scores",
    )
    parser.add_argument(
        "--skip-receiver",
        action="store_true",
        help="Skip caching receiver head scores",
    )
    parser.add_argument(
        "--kl-divergence",
        action="store_true",
        help="Also cache KL divergences (computationally expensive)",
    )
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=4,
        help="Number of adjacent sentences to ignore in vertical score calculation",
    )
    parser.add_argument(
        "--control-depth",
        action="store_true",
        help="Control for depth in vertical score calculation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top receiver heads to identify",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (for testing)",
    )
    
    args = parser.parse_args()
    
    if args.all_models:
        models = ["qwen-15b", "qwen-14b", "llama-8b"]
    elif args.model:
        models = [args.model]
    else:
        models = ["qwen-15b", "qwen-14b"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            problems = get_all_problems(model_name)
            
            if args.max_problems:
                problems = problems[:args.max_problems]
                print(f"Limiting to first {args.max_problems} problems for testing")
            
            print(f"Found {len(problems)} problems to cache")
            
            if not args.skip_attention:
                cache_attention_matrices(model_name, problems, verbose=args.verbose)
            else:
                print("Skipping attention matrix caching")
            
            if not args.skip_vertical:
                cache_vertical_scores(
                    model_name,
                    problems,
                    proximity_ignore=args.proximity_ignore,
                    control_depth=args.control_depth,
                    verbose=args.verbose,
                )
            else:
                print("Skipping vertical score caching")
            
            if not args.skip_receiver:
                cache_receiver_head_scores(
                    model_name,
                    proximity_ignore=args.proximity_ignore,
                    control_depth=args.control_depth,
                    top_k=args.top_k,
                )
            else:
                print("Skipping receiver head score caching")
            
            if args.kl_divergence:
                cache_kl_divergences(
                    model_name,
                    problems,
                    p_nucleus=0.9999,
                    take_log=True,
                    verbose=args.verbose,
                )
            else:
                print("Skipping KL divergence caching (use --kl-divergence to enable)")
            
            print(f"\nCompleted caching for {model_name}")
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Caching complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()