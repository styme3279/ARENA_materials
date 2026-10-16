import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import torch
import gc
import traceback
from utils import get_chunk_ranges, get_chunk_token_ranges
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.inspection import permutation_importance

parser = argparse.ArgumentParser(description="Benchmark different attribution methods")
parser.add_argument("-ad", "--analysis_dir", type=str, default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95", help="Directory containing analysis results")
parser.add_argument("-od", "--output_dir", type=str, default="analysis/attribution_benchmark", help="Directory to save benchmark results")
parser.add_argument("-t", "--thinking_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Thinking model name")
parser.add_argument("-b", "--base_model", type=str, default="Qwen/Qwen2.5-14B", help="Base model name")
parser.add_argument("-mp", "--max_problems", type=int, default=None, help="Maximum number of problems to analyze")
parser.add_argument("-l", "--layer", type=int, default=47, help="Layer to analyze for KL attribution")
parser.add_argument("-co", "--correct_only", action="store_true", default=True, help="Only analyze correct solutions")
parser.add_argument("-uai", "--use_abs_importance", action="store_true", default=True, help="Use absolute importance values")
parser.add_argument("-it", "--importance_threshold", type=float, default=None, help="Importance threshold for chunks")
parser.add_argument("-tm", "--target_module", type=str, default='all', choices=['attention', 'mlp', 'all'], help='Which module to use for KL attribution (attention, mlp, or all)')
args = parser.parse_args()

N_ESTIMATORS = 32
METHOD = "random_forest"

def get_model():
    if METHOD == "lasso":
        return Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', Lasso(alpha=0.0001, max_iter=10000, random_state=42))
        ])
    elif METHOD == "random_forest":
        return Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=100))
        ])
    elif METHOD == "xgboost":
        return Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', xgboost.XGBRegressor(n_estimators=N_ESTIMATORS, eta=0.003, max_depth=3, random_state=42, n_jobs=100))
        ])
    elif METHOD == "elasticnet":
        return Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', ElasticNet(alpha=0.002, l1_ratio=0.9, max_iter=10000, random_state=42))
        ])
    elif METHOD == "mlp":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=(128,),  # Smaller network
                alpha=0.002,                   # Stronger regularization
                learning_rate_init=0.0001,     # Smaller learning rate
                max_iter=500,                 # Fewer iterations to prevent overfitting
                early_stopping=True,          # Use early stopping
                validation_fraction=0.1,      # Validation set size
                n_iter_no_change=10,          # Patience for early stopping
                random_state=42,
                solver='adam'                 # Adam optimizer works well
            ))
        ])

def get_mlp_feature_importance(model, X):
    """
    Calculate feature importance for MLPRegressor using permutation importance.
    
    Args:
        model: Trained MLPRegressor model
        X: Feature matrix used for training
        
    Returns:
        Array of feature importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # Use permutation importance
    result = permutation_importance(
        model, X, y=None, n_repeats=10, random_state=42, 
        scoring=None  # Uses model's default scorer (R²)
    )
    
    return result.importances_mean

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Global font size for plots
FONT_SIZE = 15

# Set font size for all plots
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
    'figure.titlesize': FONT_SIZE + 4
})

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, AutoTokenizer]:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of model and tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        # Import bitsandbytes for quantization
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            attn_implementation="eager",
        )
        print("Model loaded with 4-bit quantization")
        
    except ImportError:
        print("bitsandbytes not installed. Falling back to 16-bit precision.")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager"
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def get_problem_dirs(analysis_dir: Path, correct_only: bool = True, limit: Optional[int] = None) -> List[Path]:
    """
    Get problem directories in the analysis directory that have complete chunk folders.
    
    Args:
        analysis_dir: Path to the analysis directory
        correct_only: Whether to only include correct solution directories
        limit: Optional limit on number of directories to return
        
    Returns:
        List of problem directory paths with complete chunk folders
    """
    # Determine which subdirectory to use based on correct_only
    subdir = "correct_base_solution" if correct_only else "incorrect_base_solution"
    base_dir = analysis_dir / subdir
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return []
    
    # Get all problem directories
    all_problem_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    # Filter for directories with complete chunk folders
    complete_problem_dirs = []
    incomplete_problem_dirs = []
    no_chunks_problem_dirs = []
    
    for problem_dir in all_problem_dirs:
        # Check if chunks_labeled.json exists
        chunks_file = problem_dir / "chunks_labeled.json"
        if not chunks_file.exists():
            no_chunks_problem_dirs.append(problem_dir)
            continue
        
        # Load chunks data
        with open(chunks_file, 'r', encoding='utf-8') as f:
            try:
                chunks_data = json.load(f)
            except json.JSONDecodeError:
                no_chunks_problem_dirs.append(problem_dir)
                continue
        
        # Check if all chunks have folders
        all_chunks_have_folders = True
        for chunk in chunks_data:
            chunk_idx = chunk.get("chunk_idx", -1)
            if chunk_idx == -1:
                all_chunks_have_folders = False
                break
                
            chunk_dir = problem_dir / f"chunk_{chunk_idx}"
            if not chunk_dir.exists() or not chunk_dir.is_dir():
                all_chunks_have_folders = False
                break
        
        if all_chunks_have_folders:
            complete_problem_dirs.append(problem_dir)
        else:
            incomplete_problem_dirs.append(problem_dir)
    
    # Print statistics
    total_problems = len(all_problem_dirs)
    complete_count = len(complete_problem_dirs)
    incomplete_count = len(incomplete_problem_dirs)
    no_chunks_count = len(no_chunks_problem_dirs)
    
    print(f"Found {total_problems} problem directories in {base_dir}")
    print(f"  - {complete_count} ({complete_count/total_problems*100:.1f}%) have complete chunk folders")
    print(f"  - {incomplete_count} ({incomplete_count/total_problems*100:.1f}%) have incomplete chunk folders")
    print(f"  - {no_chunks_count} ({no_chunks_count/total_problems*100:.1f}%) have no chunks data")
    
    # Apply limit if specified
    if limit and limit < len(complete_problem_dirs):
        complete_problem_dirs = complete_problem_dirs[:limit]
        print(f"Limited to {limit} problem directories")
    
    return complete_problem_dirs

def load_labeled_chunks(problem_dir: Path) -> List[Dict]:
    """
    Load labeled chunks from a problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        
    Returns:
        List of labeled chunks with importance scores
    """
    chunks_file = problem_dir / "chunks_labeled.json"
    
    if not chunks_file.exists():
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        
    # Augment chunks with importance scores facotring in use_abs_importance
    for chunk in chunks_data:
        if args.use_abs_importance:
            chunk["importance"] = abs(chunk["importance"])
        if args.importance_threshold:
            chunk["importance"] = chunk["importance"] if chunk["importance"] >= args.importance_threshold else 0
    
    return chunks_data

def get_base_solution(problem_dir: Path) -> Dict:
    """
    Get base solution from a problem directory.
    """
    base_solution_path = problem_dir / "base_solution.json"
    if not base_solution_path.exists():
        return None
    
    return json.load(open(base_solution_path, 'r', encoding='utf-8'))

def load_problem(problem_dir: Path) -> Dict:
    """
    Load problem from a problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        
    Returns:
        Problem dictionary
    """
    problem_file = problem_dir / "problem.json"
    
    if not problem_file.exists():
        return {}
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    
    return problem_data

def llm_attribution(problem: Dict, chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use GPT-4o to rate chunks by importance in a single call.
    
    Args:
        problem: Problem dictionary
        chunks: List of chunk dictionaries
        base_solution: Base solution dictionary
        **kwargs: Additional arguments
        
    Returns:
        List of normalized importance scores (0-1 range)
    """
    # Extract problem text and chunks
    problem_text = problem.get("problem", "")
    chunk_texts = [chunk.get("chunk", "") for chunk in chunks]
    
    # Create the prompt with all chunks
    prompt = f"""
    You are analyzing a mathematical problem and its solution broken into chunks. 
    Your task is to determine the importance of each chunk to the solution.

    Problem:
    {problem_text}

    I need you to provide ONLY a comma-separated list of chunk numbers, ordered from most important to least important with respect to the solution. 
    The most important chunk should be first, and the least important chunk should be last.
    
    For example, if there are 5 chunks and chunk 3 is most important, followed by chunk 1, then chunk 5, then chunk 2, 
    and chunk 4 is least important, your response should look exactly like this:
    3, 1, 5, 2, 4
    
    Please put a comma and a space between each chunk number.
    
    Do not include any explanations, ratings, or other text. Just the ordered list of chunk numbers.
    
    If there are 100 chunks, there should be 100 numbers separated by commas in your response. Do NOT skip any chunks!
    
    VERY IMPORTANT: Do not duplicate chunk numbers! Chunk numbers should be unique. If you're given 100 chunks, there MUST be 100 numbers in your response. No more, no less.

    Solution chunks to analyze:
    """
    
    # Add each chunk with its index
    for i, chunk_text in enumerate(chunk_texts):
        prompt += f"\n{i+1}. {chunk_text}"
    
    try:
        # Make a single call to the LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096
        )
        
        # Extract response
        response_text = response.choices[0].message.content.strip().replace(".", "")
        print('LLM response:', response_text)
        
        # Parse the ordered indices
        try:
            # Split by commas and convert to integers
            ordered_indices = [int(x.strip()) for x in response_text.split(',')]
            
            # Convert to 0-based indices if they're 1-based
            if all(n >= 1 for n in ordered_indices):
                ordered_indices = [int(n) - 1 for n in ordered_indices]
                
            # Verify we have the right number of indices
            if len(ordered_indices) != len(chunk_texts):
                print(f"Warning: Got {len(ordered_indices)} indices but expected {len(chunk_texts)}.")
                
                # If we have too few, append missing indices in arbitrary order
                if len(ordered_indices) < len(chunk_texts):
                    missing_indices = set(range(len(chunk_texts))) - set(ordered_indices)
                    ordered_indices.extend(list(missing_indices))
                    
                # If we have too many, truncate
                if len(ordered_indices) > len(chunk_texts):
                    ordered_indices = ordered_indices[:len(chunk_texts)]
                    
            # Convert ordering to scores (higher score = more important)
            scores = [0.0] * len(chunk_texts)
            for rank, chunk_idx in enumerate(ordered_indices):
                if 0 <= chunk_idx < len(chunk_texts):  # Ensure index is valid
                    # Reverse the rank (so highest rank = highest score)
                    scores[chunk_idx] = len(chunk_texts) - rank
                    
            # Normalize scores to 0-1 range
            max_score = max(scores)
            min_score = min(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.5] * len(scores)
                
            return normalized_scores
            
        except Exception as e:
            print(f"Error parsing ordered indices: {e}")
            traceback.print_exc()
            return [0.5] * len(chunk_texts)  # Default to middle scores
        
    except Exception as e:
        print(f"Error getting LLM attribution: {e}")
        traceback.print_exc()
        return [0.5] * len(chunk_texts)  # Default to middle scores

def kl_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Compute KL divergence attribution for chunks.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - thinking_model: The thinking model
            - base_model: The base model
            - tokenizer: Tokenizer for both models
            - layer_idx: Layer index to analyze
        
    Returns:
        List of attribution scores (higher = more important)
    """
    thinking_model = kwargs.get("thinking_model")
    base_model = kwargs.get("base_model")
    tokenizer = kwargs.get("tokenizer")
    layer_idx = kwargs.get("layer_idx", 47)
    
    if not thinking_model or not base_model or not tokenizer:
        print("Missing required models or tokenizer for KL attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(thinking_model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get base model logits (no gradients)
        with torch.no_grad():
            base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs.logits
            
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get thinking model logits
        with torch.no_grad():
            thinking_outputs = thinking_model(input_ids=input_ids, attention_mask=attention_mask)
            thinking_logits = thinking_outputs.logits
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Compute KL divergence for each token position
        kl_divergences = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            base_next_token_logits = base_logits[:, pos, :]
            thinking_next_token_logits = thinking_logits[:, pos, :]
            
            # Convert to probabilities
            base_probs = torch.nn.functional.softmax(base_next_token_logits, dim=-1)
            base_log_probs = torch.nn.functional.log_softmax(base_next_token_logits, dim=-1)
            thinking_log_probs = torch.nn.functional.log_softmax(thinking_next_token_logits, dim=-1)
            
            kl_div = (base_probs * (base_log_probs - thinking_log_probs)).sum(dim=-1).item()
            kl_divergences.append(kl_div)
        
        # Compute attribution scores for each chunk by averaging KL divergence over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(kl_divergences) - 1)
            end_idx = min(end_idx, len(kl_divergences))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average KL divergence over chunk tokens
            chunk_kl = np.mean(kl_divergences[start_idx:end_idx])
            attribution_scores.append(float(chunk_kl))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN values with zeros
        attribution_scores = [0.0 if np.isnan(score) else score for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in KL attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def entropy_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use token entropy to measure chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - model: The model
            - tokenizer: The tokenizer
        
    Returns:
        List of attribution scores (higher = more important)
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    
    if not model or not tokenizer:
        print("Missing required model or tokenizer for entropy attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate entropy for each token position
        token_entropies = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            next_token_logits = logits[:, pos, :]
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1).item()
            token_entropies.append(entropy)
        
        # Compute attribution scores for each chunk by averaging entropy over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(token_entropies) - 1)
            end_idx = min(end_idx, len(token_entropies))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average entropy over chunk tokens
            chunk_entropy = np.mean(token_entropies[start_idx:end_idx])
            attribution_scores.append(float(chunk_entropy))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN values with zeros
        attribution_scores = [0.0 if np.isnan(score) else score for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in entropy attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def perplexity_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use token perplexity to measure chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including:
            - model: The model
            - tokenizer: The tokenizer
        
    Returns:
        List of attribution scores (higher = more important)
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    
    if not model or not tokenizer:
        print("Missing required model or tokenizer for perplexity attribution")
        return [0.0] * len(labeled_chunks)
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return [0.0] * len(labeled_chunks)
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate perplexity for each token position
        token_perplexities = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            next_token_logits = logits[:, pos, :]
            
            # Get the target token
            target_id = input_ids[:, pos + 1]
            
            # Calculate loss for this token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(next_token_logits, target_id)
            
            # Calculate perplexity: exp(loss)
            perplexity = torch.exp(loss).item()
            # Cap perplexity to avoid extreme values
            perplexity = min(perplexity, 1000.0)
            token_perplexities.append(perplexity)
        
        # Compute attribution scores for each chunk by averaging perplexity over chunk tokens
        attribution_scores = []
        
        for start_idx, end_idx in chunk_token_ranges:
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(token_perplexities) - 1)
            end_idx = min(end_idx, len(token_perplexities))
            
            if start_idx >= end_idx:
                attribution_scores.append(0.0)
                continue
            
            # Average perplexity over chunk tokens
            chunk_perplexity = np.mean(token_perplexities[start_idx:end_idx])
            attribution_scores.append(float(chunk_perplexity))
        
        # If we couldn't match some chunks, fill with zeros
        if len(attribution_scores) < len(labeled_chunks):
            attribution_scores.extend([0.0] * (len(labeled_chunks) - len(attribution_scores)))
        
        # Replace NaN and inf values with zeros or max value
        attribution_scores = [0.0 if np.isnan(score) else (1000.0 if np.isinf(score) else score) for score in attribution_scores]
        
        return attribution_scores
    
    except Exception as e:
        print(f"Error in perplexity attribution: {e}")
        traceback.print_exc()
        return [0.0] * len(labeled_chunks)

def token_length_attribution(problem: Dict, chunks: List[Dict], base_solution: Dict, **kwargs) -> List[int]:
    """
    Use token length to rank chunks by importance.
    
    Args:
        problem: Problem dictionary
        chunks: List of chunk dictionaries
        base_solution: Base solution dictionary
        **kwargs: Additional arguments
        
    Returns:
        List of ranks for each chunk (lower is more important)
    """
    # Extract chunk texts
    chunk_texts = [chunk.get("chunk", "") for chunk in chunks]
    
    # Calculate token length for each chunk
    tokenizer = kwargs.get("tokenizer")
    
    if tokenizer:
        # Use the provided tokenizer
        token_lengths = [len(tokenizer.encode(chunk)) for chunk in chunk_texts]
    else:
        # Approximate token length
        token_lengths = [len(chunk.split()) for chunk in chunk_texts]
    
    # Convert lengths to ranks (longer = more important = lower rank)
    ranks = [0] * len(chunks)
    sorted_indices = np.argsort(token_lengths)[::-1]  # Sort in descending order
    
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    
    return ranks

def evaluate_attribution_method(
    problem_dirs: List[Path],
    attribution_method: Callable,
    method_name: str,
    **kwargs
) -> Dict:
    """
    Evaluate an attribution method on multiple problems.
    
    Args:
        problem_dirs: List of problem directories
        attribution_method: Function that implements the attribution method
        method_name: Name of the attribution method
        **kwargs: Additional arguments to pass to the attribution method
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {method_name} attribution method...")
    
    correlations = []
    
    for problem_dir in tqdm(problem_dirs, desc=f"Evaluating {method_name}"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        base_solution = get_base_solution(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Load problem
        problem = load_problem(problem_dir)
        
        if not problem:
            continue
        
        # Extract true importance scores
        true_scores = [np.abs(chunk.get("importance", 0.0)) if kwargs.get("use_abs_importance") else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Skip if all scores are the same
        if len(set(true_scores)) <= 1:
            continue
        
        # Normalize true scores to 0-1 range
        min_true = min(true_scores)
        max_true = max(true_scores)
        if max_true > min_true:
            true_scores = [(s - min_true) / (max_true - min_true) for s in true_scores]
        
        # Get predicted scores
        predicted_scores = attribution_method(problem, labeled_chunks, base_solution, **kwargs)
        
        # Normalize predicted scores to 0-1 range
        if predicted_scores:
            min_pred = min(predicted_scores)
            max_pred = max(predicted_scores)
            if max_pred > min_pred:
                predicted_scores = [(s - min_pred) / (max_pred - min_pred) for s in predicted_scores]
        
        # Skip if lengths don't match
        if len(predicted_scores) != len(true_scores):
            continue
        
        # Calculate correlation
        correlation = np.corrcoef(true_scores, predicted_scores)[0, 1]
        if not np.isnan(correlation):
            correlations.append(correlation)
    
    # Calculate average correlation
    avg_correlation = np.mean(correlations) if correlations else 0.0
    
    return {
        "method": method_name,
        "correlation": avg_correlation,
        "num_problems": len(correlations)
    }

def plot_results(results: List[Dict], output_dir: Path) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: List of evaluation results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by correlation
    df = df.sort_values("correlation", ascending=False)
    
    # Create correlation plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="method", y="correlation", data=df)
    plt.title("Correlation with Ground Truth by Attribution Method")
    plt.xlabel("Method")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation.png")
    plt.close()
    
    # Save results to CSV
    df.to_csv(output_dir / "attribution_results.csv", index=False)
    
    print(f"Results saved to {output_dir}")

def plot_attribution_metrics(
    problem_dirs: List[Path],
    attribution_methods: Dict[str, Callable],
    output_dir: Path,
    supervised_methods: Dict[str, Callable] = None,
    **kwargs
) -> None:
    """
    Plot attribution metrics alongside ground truth importance for each problem.
    
    Args:
        problem_dirs: List of problem directories
        attribution_methods: Dictionary mapping method names to attribution functions
        output_dir: Directory to save plots
        supervised_methods: Dictionary of supervised methods that need leave-one-out training
        **kwargs: Additional arguments to pass to attribution methods
    """
    print("Plotting attribution metrics for each problem...")

    # Create plots directory
    plots_dir = output_dir / "attribution_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    use_abs_importance = kwargs.get("use_abs_importance", True)
    
    # Make a copy of kwargs without the models to avoid duplicate keyword arguments
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in [
        'activation_model', 'metrics_model', 'activation_position_model', 
        'metrics_position_model', 'position_model'
    ]}
    
    for problem_dir in tqdm(problem_dirs, desc="Plotting attribution metrics"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        base_solution = get_base_solution(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Load problem
        problem = load_problem(problem_dir)
        
        if not problem:
            continue
        
        # Extract problem index from directory name
        problem_idx = problem_dir.name.split('_')[-1]
        
        # Extract true importance scores
        true_scores = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Skip if all scores are the same
        if len(set(true_scores)) <= 1:
            continue
        
        # Get attribution scores for unsupervised methods
        method_scores = {}
        for method_name, attribution_method in attribution_methods.items():
            # Skip "All Data" methods for plots
            if "(All Data)" in method_name:
                continue
                
            try:
                scores = attribution_method(problem, labeled_chunks, base_solution, **plot_kwargs)
                method_scores[method_name] = scores
            except Exception as e:
                print(f"Error computing {method_name} for problem {problem_idx}: {e}")
                traceback.print_exc()
        
        # For supervised methods, train on all problems except this one
        if supervised_methods:
            other_problem_dirs = [dir for dir in problem_dirs if dir != problem_dir]
            
            for method_name, method_factory in supervised_methods.items():
                try:
                    # Train model on all other problems
                    if method_name == "Activation-Regression":
                        model = train_regression_model(
                            other_problem_dirs,
                            model_type="activation",
                            **kwargs
                        )
                        if model:
                            # Get predictions using this model
                            scores = method_factory(model)(problem, labeled_chunks, base_solution, **plot_kwargs)
                            method_scores[f"{method_name}"] = scores
                    elif method_name == "Metrics-Regression":
                        model = train_regression_model(
                            other_problem_dirs,
                            model_type="metrics",
                            **kwargs
                        )
                        if model:
                            # Get predictions using this model
                            scores = method_factory(model)(problem, labeled_chunks, base_solution, **plot_kwargs)
                            method_scores[f"{method_name}"] = scores
                    elif method_name == "Activation-Position-Regression":
                        model = train_regression_model(
                            other_problem_dirs,
                            model_type="activation_position",
                            **kwargs
                        )
                        if model:
                            # Get predictions using this model
                            scores = method_factory(model)(problem, labeled_chunks, base_solution, **plot_kwargs)
                            method_scores[f"{method_name}"] = scores
                    elif method_name == "Metrics-Position-Regression":
                        model = train_regression_model(
                            other_problem_dirs,
                            model_type="metrics_position",
                            **kwargs
                        )
                        if model:
                            # Get predictions using this model
                            scores = method_factory(model)(problem, labeled_chunks, base_solution, **plot_kwargs)
                            method_scores[f"{method_name}"] = scores
                    elif method_name == "Position-Regression":
                        model = train_position_regression_model(
                            other_problem_dirs,
                            **kwargs
                        )
                        if model:
                            # Get predictions using this model
                            scores = method_factory(model)(problem, labeled_chunks, base_solution, **plot_kwargs)
                            method_scores[f"{method_name}"] = scores
                except Exception as e:
                    print(f"Error with LOOCV for {method_name} on problem {problem_idx}: {e}")
                    traceback.print_exc()
        
        # Skip if no methods produced scores
        if not method_scores:
            continue
        
        # Create subplot structure - one subplot per method plus ground truth
        num_methods = len(method_scores)
        fig, axes = plt.subplots(num_methods + 1, 1, figsize=(12, 3 * (num_methods + 1)), sharex=True)
        
        # Plot ground truth importance in the first subplot
        axes[0].plot(range(len(true_scores)), true_scores, 'k-', linewidth=2)
        axes[0].set_title("Ground Truth Importance")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel("Score")
        
        # Plot each attribution method in its own subplot
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        
        for i, (method_name, scores) in enumerate(method_scores.items()):
            if len(scores) != len(true_scores):
                print(f"Warning: {method_name} scores length ({len(scores)}) doesn't match true scores length ({len(true_scores)}) for problem {problem_idx}")
                continue
                
            color = colors[i % len(colors)]
            
            # Normalize scores to 0-1 range for better visualization
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.5] * len(scores)
            
            # Plot in the corresponding subplot
            axes[i+1].plot(range(len(scores)), normalized_scores, f'{color}-', linewidth=1.5)
            axes[i+1].set_title(f"{method_name}")
            axes[i+1].grid(True, alpha=0.3)
            axes[i+1].set_ylabel("Score (normalized)")
        
        # Set common x-axis label
        axes[-1].set_xlabel("Chunk Index")
        
        # Add overall title
        plt.suptitle(f"Problem {problem_idx}: Attribution Metrics vs Ground Truth", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the plot
        plt.savefig(plots_dir / f"problem_{problem_idx}_attribution.png")
        plt.close()
    
    print(f"Attribution metric plots saved to {plots_dir}")

def get_chunk_features(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> pd.DataFrame:
    """
    Extract features for each chunk to be used in regression models.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including models and tokenizer
        
    Returns:
        DataFrame with features for each chunk
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    thinking_model = kwargs.get("thinking_model")
    base_model = kwargs.get("base_model")
    layer_idx = kwargs.get("layer_idx", 47)
    
    if not model or not tokenizer:
        print("Missing required model or tokenizer for feature extraction")
        return pd.DataFrame()
    
    try:
        # Get the full CoT
        full_cot = base_solution.get("full_cot", "")
        if not full_cot:
            print(f"No full CoT found in base solution")
            return pd.DataFrame()
        
        # Extract chunks from labeled_chunks
        chunks = [chunk.get("chunk", "") for chunk in labeled_chunks]
        
        # Get character ranges for each chunk in the full CoT
        chunk_ranges = get_chunk_ranges(full_cot, chunks)
        
        # Convert character ranges to token ranges
        chunk_token_ranges = get_chunk_token_ranges(full_cot, chunk_ranges, tokenizer)
        
        # Tokenize the full CoT
        device = next(model.parameters()).device
        inputs = tokenizer(full_cot, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Set up hooks to capture hidden states from the specified layer
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                # Handle case where output is a tuple (like in Qwen2)
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hook for the specified layer
        hooks = []
        try:
            # Try different model architectures
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # LLaMA, Mistral, etc.
                target_layer = model.model.layers[layer_idx]
                hooks.append(target_layer.register_forward_hook(get_activation(f"layer_{layer_idx}")))
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # GPT-2 style
                target_layer = model.transformer.h[layer_idx]
                hooks.append(target_layer.register_forward_hook(get_activation(f"layer_{layer_idx}")))
            elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                # T5 style
                target_layer = model.model.decoder.layers[layer_idx]
                hooks.append(target_layer.register_forward_hook(get_activation(f"layer_{layer_idx}")))
            else:
                print(f"Could not find layer {layer_idx} in model architecture")
        except (AttributeError, IndexError) as e:
            print(f"Error accessing layer {layer_idx}: {e}")
        
        # Get model outputs for entropy and perplexity
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get KL divergence features if both models are provided
        kl_divergences = []
        if thinking_model and base_model:
            with torch.no_grad():
                base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = base_outputs.logits
                
                # Calculate KL divergence for each token position
                for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
                    # Get logits for the next token
                    base_next_token_logits = base_logits[:, pos, :]
                    thinking_next_token_logits = logits[:, pos, :]
                    
                    # Convert to probabilities
                    base_probs = torch.nn.functional.softmax(base_next_token_logits, dim=-1)
                    base_log_probs = torch.nn.functional.log_softmax(base_next_token_logits, dim=-1)
                    thinking_log_probs = torch.nn.functional.log_softmax(thinking_next_token_logits, dim=-1)
                    
                    kl_div = (base_probs * (base_log_probs - thinking_log_probs)).sum(dim=-1).item()
                    # Clip KL divergence to avoid extreme values
                    kl_div = max(min(kl_div, 100.0), -100.0)
                    kl_divergences.append(kl_div)
        
        # Calculate entropy for each token position
        token_entropies = []
        token_perplexities = []
        
        for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
            # Get logits for the next token
            next_token_logits = logits[:, pos, :]
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1).item()
            # Clip entropy to avoid extreme values
            entropy = max(min(entropy, 100.0), 0.0)
            token_entropies.append(entropy)
            
            # Get the target token
            target_id = input_ids[:, pos + 1]
            
            # Calculate loss for this token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(next_token_logits, target_id)
            
            # Calculate perplexity: exp(loss)
            perplexity = torch.exp(loss).item()
            # Cap perplexity to avoid extreme values
            perplexity = min(perplexity, 1000.0)
            token_perplexities.append(perplexity)
        
        # Extract features for each chunk
        features = []
        
        for i, (start_idx, end_idx) in enumerate(chunk_token_ranges):
            # Ensure indices are within bounds
            start_idx = min(start_idx, len(token_entropies) - 1)
            end_idx = min(end_idx, len(token_entropies))
            
            if start_idx >= end_idx:
                # Skip this chunk if indices are invalid
                continue
            
            # Basic features
            chunk_features = {
                # 'chunk_idx': i,
                # 'position': i / len(chunk_token_ranges),  # Normalized position (0-1)
                # 'relative_position': i / (len(chunk_token_ranges) - 1) if len(chunk_token_ranges) > 1 else 0.5,  # Normalized position (0-1)
                # 'token_length': end_idx - start_idx,
                # 'char_length': len(chunks[i]),
                # 'word_length': len(chunks[i].split()),
            }
            
            # Entropy features
            if token_entropies:
                chunk_entropies = token_entropies[start_idx:end_idx]
                chunk_features.update({
                    'mean_entropy': np.mean(chunk_entropies),
                    'max_entropy': np.max(chunk_entropies),
                    'min_entropy': np.min(chunk_entropies),
                    'std_entropy': np.std(chunk_entropies),
                })
            
            # Perplexity features
            if token_perplexities:
                chunk_perplexities = token_perplexities[start_idx:end_idx]
                chunk_features.update({
                    'mean_perplexity': np.mean(chunk_perplexities),
                    'max_perplexity': np.max(chunk_perplexities),
                    'min_perplexity': np.min(chunk_perplexities),
                    'std_perplexity': np.std(chunk_perplexities),
                })
            
            # KL divergence features
            if kl_divergences:
                chunk_kl = kl_divergences[start_idx:end_idx]
                chunk_features.update({
                    'mean_kl': np.mean(chunk_kl),
                    'max_kl': np.max(chunk_kl),
                    'min_kl': np.min(chunk_kl),
                    'std_kl': np.std(chunk_kl),
                })
            
            # Activation features from hidden states
            layer_key = f"layer_{layer_idx}"
            if layer_key in activations:
                layer_activations = activations[layer_key]
                chunk_activations = layer_activations[0, start_idx:end_idx, :].detach().cpu().numpy()
                
                # Compute mean activation per dimension (first N dimensions only)
                mean_activations = np.mean(chunk_activations, axis=0)
                for i in range(len(mean_activations)):  # Use only first 10 dimensions
                    chunk_features[f'mean_activation_dim_{i}'] = float(mean_activations[i])
                
                # Add some principal components of activations
                """
                try:
                    from sklearn.decomposition import PCA
                    if chunk_activations.shape[0] > 1:  # Need at least 2 tokens for PCA
                        pca = PCA(n_components=min(5, chunk_activations.shape[0]))
                        pca_result = pca.fit_transform(chunk_activations)
                        for j in range(pca_result.shape[1]):
                            chunk_features[f'pca_{j}'] = np.mean(pca_result[:, j])
                except Exception as e:
                    print(f"Error computing PCA: {e}")
                """
            
            features.append(chunk_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def train_regression_model(problem_dirs: List[Path], model_type: str, **kwargs) -> Any:
    """
    Train a regression model to predict chunk importance.
    
    Args:
        problem_dirs: List of problem directories
        model_type: Type of model to train ('activation' or 'metrics')
        **kwargs: Additional arguments
        
    Returns:
        Trained model
    """
    print(f"Training {model_type} regression model...")
    
    # Collect features and labels from all problems
    all_features = []
    all_labels = []
    
    for problem_dir in tqdm(problem_dirs, desc=f"Collecting {model_type} features"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        base_solution = get_base_solution(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Load problem
        problem = load_problem(problem_dir)
        
        if not problem:
            continue
        
        # Get features
        features_df = get_chunk_features(problem, labeled_chunks, base_solution, **kwargs)
        
        if features_df.empty:
            continue
        
        # Add position feature
        num_chunks = len(labeled_chunks)
        if num_chunks > 1:
            features_df['position'] = [i / (num_chunks - 1) for i in range(num_chunks)]
            features_df['chunk_index'] = [i for i in range(num_chunks)]
        else:
            features_df['position'] = [0.5]
            features_df['chunk_index'] = [0]
            
        # Add importance label
        use_abs_importance = kwargs.get("use_abs_importance", True)
        features_df['importance'] = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Select features based on model type
        if model_type == "activation":
            # Use activation-based features
            selected_cols = [col for col in features_df.columns if 
                            ('activation' in col or 'pca_' in col)]
        elif model_type == "metrics":
            # Use metrics-based features
            selected_cols = [col for col in features_df.columns if 
                            ('entropy' in col or 'perplexity' in col or 'kl' in col)]
        elif model_type == "activation_position":
            # Use activation-based features plus position
            selected_cols = [col for col in features_df.columns if 
                            ('activation' in col or 'pca_' in col or 
                             col == 'position' or col == 'chunk_index')]
        elif model_type == "metrics_position":
            # Use metrics-based features plus position
            selected_cols = [col for col in features_df.columns if 
                            ('entropy' in col or 'perplexity' in col or 'kl' in col or
                             col == 'position' or col == 'chunk_index')]
        else:
            # Default to all features
            selected_cols = [col for col in features_df.columns if col != 'importance']
        
        # Ensure we have some features
        if not selected_cols:
            continue
        
        # Store the feature names for this problem
        if problem_dir == problem_dirs[0]:
            print(f"\nNum. features for {model_type} model: {len(selected_cols)}")
        
        # Add to collection
        all_features.append(features_df[selected_cols])
        all_labels.append(features_df['importance'])
    
    # Combine all data
    if not all_features:
        print(f"No valid data collected for {model_type} model")
        return None
    
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Replace any infinite values with large but finite numbers
    X = X.replace([np.inf, -np.inf], [1e6, -1e6])
    
    # Store the feature names
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create and train model
    model = get_model()
    
    try:
        model.fit(X_train, y_train)
        
        # Store the feature names in the model
        model.feature_names = feature_names
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"{model_type.capitalize()} model R² on training data: {train_score:.4f}")
        print(f"{model_type.capitalize()} model R² on test data: {test_score:.4f}")
        
        # Feature importance
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            feature_importance = model.named_steps['regressor'].feature_importances_
        elif hasattr(model.named_steps['regressor'], 'coef_'):
            feature_importance = np.abs(model.named_steps['regressor'].coef_)
        elif METHOD == "mlp":
            # For MLP, use permutation importance
            feature_importance = get_mlp_feature_importance(model, X_train)
        else:
            feature_importance = np.ones(len(feature_names))
        
        # Print top features
        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top features for {model_type} model:")
        for name, importance in top_features:
            print(f"  - {name}: {importance:.4f}")
        
        return model
    except Exception as e:
        print(f"Error training {model_type} model: {e}")
        traceback.print_exc()
        return None

def activation_regression_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use activation-based regression model to predict chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including the trained model
        
    Returns:
        List of predicted importance scores
    """
    model = kwargs.get("activation_model")
    
    if not model:
        print("Missing activation regression model")
        return [0.5] * len(labeled_chunks)
    
    try:
        # Get features
        features_df = get_chunk_features(problem, labeled_chunks, base_solution, **kwargs)
        
        if features_df.empty:
            return [0.5] * len(labeled_chunks)
        
        # Select activation-based features
        selected_cols = [col for col in features_df.columns if 
                        ('activation' in col or 'pca_' in col or 
                         col in ['position', 'token_length'])]
        
        # Ensure we have the necessary features
        if not all(col in features_df.columns for col in selected_cols):
            missing = [col for col in selected_cols if col not in features_df.columns]
            print(f"Missing features for activation regression: {missing}")
            return [0.5] * len(labeled_chunks)
        
        # Handle missing values
        features = features_df[selected_cols].fillna(0)
        
        # Predict importance
        predicted_scores = model.predict(features)
        
        # Ensure we have the right number of predictions
        if len(predicted_scores) < len(labeled_chunks):
            predicted_scores = np.append(predicted_scores, [0.5] * (len(labeled_chunks) - len(predicted_scores)))
        elif len(predicted_scores) > len(labeled_chunks):
            predicted_scores = predicted_scores[:len(labeled_chunks)]
        
        return predicted_scores.tolist()
    
    except Exception as e:
        print(f"Error in activation regression attribution: {e}")
        traceback.print_exc()
        return [0.5] * len(labeled_chunks)

def metrics_regression_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use metrics-based regression model to predict chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including the trained model
        
    Returns:
        List of predicted importance scores
    """
    model = kwargs.get("metrics_model")
    
    if not model:
        print("Missing metrics regression model")
        return [0.5] * len(labeled_chunks)
    
    try:
        # Get features
        features_df = get_chunk_features(problem, labeled_chunks, base_solution, **kwargs)
        
        if features_df.empty:
            return [0.5] * len(labeled_chunks)
        
        # Select metrics-based features
        selected_cols = [col for col in features_df.columns if 
                        ('entropy' in col or 'perplexity' in col or 'kl' in col or
                         col in ['position', 'token_length'])]
        
        # Ensure we have the necessary features
        if not all(col in features_df.columns for col in selected_cols):
            missing = [col for col in selected_cols if col not in features_df.columns]
            print(f"Missing features for metrics regression: {missing}")
            return [0.5] * len(labeled_chunks)
        
        # Handle missing values
        features = features_df[selected_cols].fillna(0)
        
        # Predict importance
        predicted_scores = model.predict(features)
        
        # Ensure we have the right number of predictions
        if len(predicted_scores) < len(labeled_chunks):
            predicted_scores = np.append(predicted_scores, [0.5] * (len(labeled_chunks) - len(predicted_scores)))
        elif len(predicted_scores) > len(labeled_chunks):
            predicted_scores = predicted_scores[:len(labeled_chunks)]
        
        return predicted_scores.tolist()
    
    except Exception as e:
        print(f"Error in metrics regression attribution: {e}")
        traceback.print_exc()
        return [0.5] * len(labeled_chunks)

def evaluate_attribution_methods_loocv(
    problem_dirs: List[Path],
    supervised_methods: Dict[str, Callable],
    unsupervised_methods: Dict[str, Callable],
    **kwargs
) -> List[Dict]:
    """
    Evaluate attribution methods using leave-one-out cross-validation for supervised methods.
    
    Args:
        problem_dirs: List of problem directories
        supervised_methods: Dictionary mapping method names to attribution method factories
        unsupervised_methods: Dictionary mapping method names to attribution methods
        **kwargs: Additional arguments to pass to attribution methods
        
    Returns:
        List of dictionaries with evaluation results
    """
    results = []
    
    # Evaluate unsupervised methods
    for method_name, attribution_method in unsupervised_methods.items():
        print(f"Evaluating {method_name}...")
        
        # Collect correlations for each problem
        correlations = []
        
        for problem_dir in tqdm(problem_dirs, desc=f"Evaluating {method_name}"):
            # Load labeled chunks
            labeled_chunks = load_labeled_chunks(problem_dir)
            base_solution = get_base_solution(problem_dir)
            
            if not labeled_chunks:
                continue
            
            # Load problem
            problem = load_problem(problem_dir)
            
            if not problem:
                continue
            
            # Extract true importance scores
            use_abs_importance = kwargs.get("use_abs_importance", True)
            true_scores = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in labeled_chunks]
            
            # Skip if all scores are the same
            if len(set(true_scores)) <= 1:
                continue
            
            # Get attribution scores
            try:
                scores = attribution_method(problem, labeled_chunks, base_solution, **kwargs)
                
                # Compute correlation
                correlation = np.corrcoef(true_scores, scores)[0, 1]
                
                # Skip NaN correlations
                if not np.isnan(correlation):
                    correlations.append(correlation)
            except Exception as e:
                print(f"Error evaluating {method_name} on problem {problem_dir.name}: {e}")
                traceback.print_exc()
        
        # Compute average correlation
        if correlations:
            avg_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)
            
            results.append({
                "method": method_name,
                "correlation": avg_correlation,
                "std": std_correlation,
                "n": len(correlations)
            })
            
            print(f"{method_name}: Average correlation = {avg_correlation:.4f} ± {std_correlation:.4f} (n={len(correlations)})")
        else:
            print(f"{method_name}: No valid correlations")
    
    # Evaluate supervised methods using LOOCV
    for method_name, method_factory in supervised_methods.items():
        print(f"Evaluating {method_name} with LOOCV...")
        
        # Collect correlations for each problem
        correlations = []
        
        # For each problem, train on all other problems and test on this one
        for test_idx, test_problem_dir in enumerate(tqdm(problem_dirs, desc=f"LOOCV for {method_name}")):
            # Create training set (all problems except the test problem)
            train_problem_dirs = [dir_path for i, dir_path in enumerate(problem_dirs) if i != test_idx]
            
            # Train model on training set
            if method_name == "Activation-Regression":
                model = train_regression_model(
                    train_problem_dirs,
                    model_type="activation",
                    **kwargs
                )
            elif method_name == "Metrics-Regression":
                model = train_regression_model(
                    train_problem_dirs,
                    model_type="metrics",
                    **kwargs
                )
            elif method_name == "Activation-Position-Regression":
                model = train_regression_model(
                    train_problem_dirs,
                    model_type="activation_position",
                    **kwargs
                )
            elif method_name == "Metrics-Position-Regression":
                model = train_regression_model(
                    train_problem_dirs,
                    model_type="metrics_position",
                    **kwargs
                )
            elif method_name == "Position-Regression":
                model = train_position_regression_model(
                    train_problem_dirs,
                    **kwargs
                )
            else:
                print(f"Unknown supervised method: {method_name}")
                continue
            
            # Skip if model training failed
            if model is None:
                continue
            
            # Test on the held-out problem
            test_problem = load_problem(test_problem_dir)
            test_chunks = load_labeled_chunks(test_problem_dir)
            test_solution = get_base_solution(test_problem_dir)
            
            if not test_problem or not test_chunks or not test_solution:
                continue
            
            # Extract true importance scores
            use_abs_importance = kwargs.get("use_abs_importance", True)
            true_scores = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in test_chunks]
            
            # Skip if all scores are the same
            if len(set(true_scores)) <= 1:
                continue
            
            # Get attribution scores using the trained model
            try:
                attribution_method = method_factory(model)
                scores = attribution_method(test_problem, test_chunks, test_solution, **kwargs)
                
                # Compute correlation
                correlation = np.corrcoef(true_scores, scores)[0, 1]
                
                # Skip NaN correlations
                if not np.isnan(correlation):
                    correlations.append(correlation)
            except Exception as e:
                print(f"Error evaluating {method_name} on problem {test_problem_dir.name}: {e}")
                traceback.print_exc()
        
        # Compute average correlation
        if correlations:
            avg_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)
            
            results.append({
                "method": f"{method_name}",
                "correlation": avg_correlation,
                "std": std_correlation,
                "n": len(correlations)
            })
            
            print(f"{method_name} (LOOCV): Average correlation = {avg_correlation:.4f} ± {std_correlation:.4f} (n={len(correlations)})")
        else:
            print(f"{method_name} (LOOCV): No valid correlations")
    
    return results

def activation_position_regression_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use activation-based regression model with position features to predict chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including the trained model
        
    Returns:
        List of predicted importance scores
    """
    model = kwargs.get("activation_position_model")
    
    if not model:
        print("Missing activation position regression model")
        return [0.5] * len(labeled_chunks)
    
    try:
        # Get features
        features_df = get_chunk_features(problem, labeled_chunks, base_solution, **kwargs)
        
        if features_df.empty:
            return [0.5] * len(labeled_chunks)
        
        # Add position features
        num_chunks = len(labeled_chunks)
        if num_chunks > 1:
            features_df['position'] = [i / (num_chunks - 1) for i in range(num_chunks)]
        else:
            features_df['position'] = [0.5]
        
        # Select activation-based features plus position
        selected_cols = [col for col in features_df.columns if 
                        ('activation' in col or 'pca_' in col or 
                         col == 'position')]
        
        # Get the feature names the model was trained on
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
        else:
            # For pipeline, get feature names from the regressor
            if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                if hasattr(model.named_steps['regressor'], 'feature_names_in_'):
                    required_features = model.named_steps['regressor'].feature_names_in_
                else:
                    # If we can't get the feature names, use what we have
                    required_features = selected_cols
            else:
                required_features = selected_cols
        
        # Ensure we have all required features
        missing_features = [feat for feat in required_features if feat not in features_df.columns]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                features_df[feat] = 0.0
        
        # Select only the features the model was trained on
        features = features_df[required_features].fillna(0)
        
        # Replace any infinite values with large but finite numbers
        features = features.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Predict importance
        predicted_scores = model.predict(features)
        
        # Ensure we have the right number of predictions
        if len(predicted_scores) < len(labeled_chunks):
            predicted_scores = np.append(predicted_scores, [0.5] * (len(labeled_chunks) - len(predicted_scores)))
        elif len(predicted_scores) > len(labeled_chunks):
            predicted_scores = predicted_scores[:len(labeled_chunks)]
        
        return predicted_scores.tolist()
    
    except Exception as e:
        print(f"Error in activation position regression attribution: {e}")
        traceback.print_exc()
        return [0.5] * len(labeled_chunks)

def metrics_position_regression_attribution(problem: Dict, labeled_chunks: List[Dict], base_solution: Dict, **kwargs) -> List[float]:
    """
    Use metrics-based regression model with position features to predict chunk importance.
    
    Args:
        problem: Problem dictionary
        labeled_chunks: List of labeled chunks
        base_solution: Base solution dictionary
        **kwargs: Additional arguments including the trained model
        
    Returns:
        List of predicted importance scores
    """
    model = kwargs.get("metrics_position_model")
    
    if not model:
        print("Missing metrics position regression model")
        return [0.5] * len(labeled_chunks)
    
    try:
        # Get features
        features_df = get_chunk_features(problem, labeled_chunks, base_solution, **kwargs)
        
        if features_df.empty:
            return [0.5] * len(labeled_chunks)
        
        # Add position features
        num_chunks = len(labeled_chunks)
        if num_chunks > 1:
            features_df['position'] = [i / (num_chunks - 1) for i in range(num_chunks)]
        else:
            features_df['position'] = [0.5]
        
        # Select metrics-based features plus position
        selected_cols = [col for col in features_df.columns if 
                        ('entropy' in col or 'perplexity' in col or 'kl' in col or
                         col == 'position')]
        
        # Get the feature names the model was trained on
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
        else:
            # For pipeline, get feature names from the regressor
            if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                if hasattr(model.named_steps['regressor'], 'feature_names_in_'):
                    required_features = model.named_steps['regressor'].feature_names_in_
                else:
                    # If we can't get the feature names, use what we have
                    required_features = selected_cols
            else:
                required_features = selected_cols
        
        # Ensure we have all required features
        missing_features = [feat for feat in required_features if feat not in features_df.columns]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                features_df[feat] = 0.0
        
        # Select only the features the model was trained on
        features = features_df[required_features].fillna(0)
        
        # Replace any infinite values with large but finite numbers
        features = features.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Predict importance
        predicted_scores = model.predict(features)
        
        # Ensure we have the right number of predictions
        if len(predicted_scores) < len(labeled_chunks):
            predicted_scores = np.append(predicted_scores, [0.5] * (len(labeled_chunks) - len(predicted_scores)))
        elif len(predicted_scores) > len(labeled_chunks):
            predicted_scores = predicted_scores[:len(labeled_chunks)]
        
        return predicted_scores.tolist()
    
    except Exception as e:
        print(f"Error in metrics position regression attribution: {e}")
        traceback.print_exc()
        return [0.5] * len(labeled_chunks)

def train_position_regression_model(problem_dirs: List[Path], **kwargs) -> Any:
    """
    Train a regression model using only position as a feature.
    
    Args:
        problem_dirs: List of problem directories
        **kwargs: Additional arguments
        
    Returns:
        Trained regression model
    """
    print("Training position-only regression model...")
    
    # Collect features and labels
    all_features = []
    all_labels = []
    use_abs_importance = kwargs.get("use_abs_importance", True)
    
    for problem_dir in tqdm(problem_dirs, desc="Collecting data for position model"):
        # Load labeled chunks
        labeled_chunks = load_labeled_chunks(problem_dir)
        
        if not labeled_chunks:
            continue
        
        # Extract true importance scores
        true_scores = [np.abs(chunk.get("importance", 0.0)) if use_abs_importance else chunk.get("importance", 0.0) for chunk in labeled_chunks]
        
        # Skip if all scores are the same
        if len(set(true_scores)) <= 1:
            continue
        
        # Normalize true scores to 0-1 range
        min_true = min(true_scores)
        max_true = max(true_scores)
        if max_true > min_true:
            true_scores = [(s - min_true) / (max_true - min_true) for s in true_scores]
        
        # Create position features
        num_chunks = len(labeled_chunks)
        if num_chunks <= 1:
            continue
            
        positions = [i / (num_chunks - 1) for i in range(num_chunks)]
        
        # Create DataFrame with position feature
        features_df = pd.DataFrame({'position': positions})
        features_df['importance'] = true_scores
        
        # Add to collection
        all_features.append(features_df[['position']])
        all_labels.append(features_df['importance'])
    
    # Combine all data
    if not all_features:
        print("No valid data collected for position model")
        return None
    
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create and train model
    model = get_model()
    
    try:
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Position-only model R² on training data: {train_score:.4f}")
        print(f"Position-only model R² on test data: {test_score:.4f}")
        
        return model
    except Exception as e:
        print(f"Error training position model: {e}")
        traceback.print_exc()
        return None

def main():    
    # Set up directories
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(analysis_dir, correct_only=args.correct_only, limit=args.max_problems)
    
    if not problem_dirs:
        print(f"No problem directories found in {analysis_dir}")
        return
    
    print(f"Found {len(problem_dirs)} problem directories")
    
    # Load models
    print("Loading models...")
    thinking_model, tokenizer = load_model_and_tokenizer(args.thinking_model)
    base_model, _ = load_model_and_tokenizer(args.base_model)
    
    # Define unsupervised attribution methods
    unsupervised_methods = {
        # "LLM-based (GPT-4o)": llm_attribution,
        # "KL Div (wrt base model)": kl_attribution,
        # "Entropy-based": entropy_attribution,
        # "Perplexity-based": perplexity_attribution,
        # "Token-length": token_length_attribution,
    }
    
    # Define supervised attribution methods (factories that create attribution methods given a model)
    def create_activation_regression(model):
        def method(problem, chunks, solution, **kwargs):
            method_kwargs = {k: v for k, v in kwargs.items() if k != 'activation_model'}
            return activation_regression_attribution(problem, chunks, solution, activation_model=model, **method_kwargs)
        return method
    
    def create_metrics_regression(model):
        def method(problem, chunks, solution, **kwargs):
            method_kwargs = {k: v for k, v in kwargs.items() if k != 'metrics_model'}
            return metrics_regression_attribution(problem, chunks, solution, metrics_model=model, **method_kwargs)
        return method
    
    def create_activation_position_regression(model):
        def method(problem, chunks, solution, **kwargs):
            method_kwargs = {k: v for k, v in kwargs.items() if k != 'activation_position_model'}
            return activation_position_regression_attribution(problem, chunks, solution, activation_position_model=model, **method_kwargs)
        return method
    
    def create_metrics_position_regression(model):
        def method(problem, chunks, solution, **kwargs):
            method_kwargs = {k: v for k, v in kwargs.items() if k != 'metrics_position_model'}
            return metrics_position_regression_attribution(problem, chunks, solution, metrics_position_model=model, **method_kwargs)
        return method
    
    def create_position_regression(model):
        def method(problem, chunks, solution, **kwargs):
            method_kwargs = {k: v for k, v in kwargs.items() if k != 'position_model'}
            # Create a simple function that just returns the model's predictions on position features
            num_chunks = len(chunks)
            if num_chunks <= 1:
                return [0.5] * num_chunks
                
            positions = np.array([i / (num_chunks - 1) for i in range(num_chunks)]).reshape(-1, 1)
            return model.predict(positions).tolist()
        return method
    
    supervised_methods = {
        "Activation-Regression": create_activation_regression,
        # "Metrics-Regression": create_metrics_regression,
        "Activation-Position-Regression": create_activation_position_regression,
        # "Metrics-Position-Regression": create_metrics_position_regression,
        "Position-Regression": create_position_regression
    }
    
    # Evaluate all methods using LOOCV for supervised methods
    results = evaluate_attribution_methods_loocv(
        problem_dirs,
        supervised_methods,
        unsupervised_methods,
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        layer_idx=args.layer,
        target_module=args.target_module,
        use_abs_importance=args.use_abs_importance
    )
    
    # Plot results
    plot_results(results, output_dir)
    
    # For visualization, we need to train models on all data
    print("Training models on all data for visualization...")
    activation_model = train_regression_model(
        problem_dirs,
        model_type="activation",
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    
    metrics_model = None
    """
    metrics_model = train_regression_model(
        problem_dirs,
        model_type="metrics",
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    """
    
    # Train the new position-based models
    activation_position_model = train_regression_model(
        problem_dirs,
        model_type="activation_position",
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    
    metrics_position_model = None
    """
    metrics_position_model = train_regression_model(
        problem_dirs,
        model_type="metrics_position",
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    """
    
    position_model = train_position_regression_model(
        problem_dirs,
        use_abs_importance=args.use_abs_importance
    )
    
    # Plot attribution metrics for each problem
    attribution_methods = {
        # "Entropy": entropy_attribution,
        # "KL Divergence (wrt. base model)": kl_attribution,
        # "Perplexity": perplexity_attribution,
    }
    
    # Add regression methods if available
    if activation_model:
        attribution_methods["Activation-Regression (All Data)"] = lambda p, c, s, **kw: activation_regression_attribution(
            p, c, s, 
            activation_model=activation_model, 
            **{k: v for k, v in kw.items() if k != 'activation_model'}
        )
    if metrics_model:
        attribution_methods["Metrics-Regression (All Data)"] = lambda p, c, s, **kw: metrics_regression_attribution(
            p, c, s, 
            metrics_model=metrics_model, 
            **{k: v for k, v in kw.items() if k != 'metrics_model'}
        )
    if activation_position_model:
        attribution_methods["Activation-Position-Regression (All Data)"] = lambda p, c, s, **kw: activation_position_regression_attribution(
            p, c, s, 
            activation_position_model=activation_position_model, 
            **{k: v for k, v in kw.items() if k != 'activation_position_model'}
        )
    if metrics_position_model:
        attribution_methods["Metrics-Position-Regression (All Data)"] = lambda p, c, s, **kw: metrics_position_regression_attribution(
            p, c, s, 
            metrics_position_model=metrics_position_model, 
            **{k: v for k, v in kw.items() if k != 'metrics_position_model'}
        )
    if position_model:
        attribution_methods["Position-Regression (All Data)"] = lambda p, c, s, **kw: create_position_regression(position_model)(p, c, s, **kw)
    
    plot_attribution_metrics(
        problem_dirs=problem_dirs,
        attribution_methods=attribution_methods,
        supervised_methods=supervised_methods,  # Pass supervised methods for LOOCV
        output_dir=output_dir,
        model=thinking_model,
        tokenizer=tokenizer,
        thinking_model=thinking_model,
        base_model=base_model,
        activation_model=activation_model,
        metrics_model=metrics_model,
        activation_position_model=activation_position_model,
        metrics_position_model=metrics_position_model,
        position_model=position_model,
        layer_idx=args.layer,
        use_abs_importance=args.use_abs_importance
    )
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()