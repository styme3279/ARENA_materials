import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
import gc
import torch.cuda

# Set up paths
math_cots_dir = Path("math_cots")
output_dir = Path("analysis/kl_attribution")
output_dir.mkdir(exist_ok=True, parents=True)

def load_model_and_tokenizer(model_name: str, quantize: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        model_name: Name of the model to load
        quantize: Whether to quantize the model to reduce memory usage
        
    Returns:
        Tuple of model and tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if quantize and device == "cuda":
        try:
            # Import bitsandbytes for quantization
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
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
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="eager"
            )
    else:
        # Load with standard precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="eager"
        )
    
    # Set model to evaluation mode
    model.eval()
    
    print(f'Model {model_name}:')
    print('Number of layers:', model.config.num_hidden_layers)
    print('Number of attention heads:', model.config.num_attention_heads)
    print('Hidden size:', model.config.hidden_size)
    
    return model, tokenizer

def load_solutions(solutions_file: Path) -> List[Dict]:
    """
    Load solutions from a solutions file.
    
    Args:
        solutions_file: Path to the solutions file
        
    Returns:
        List of solution dictionaries
    """
    with open(solutions_file, 'r', encoding='utf-8') as f:
        solutions = json.load(f)
    
    return solutions

def get_solution_files(math_cots_dir: Path, model_name: str) -> List[Path]:
    """
    Get all solution files for a specific model.
    
    Args:
        math_cots_dir: Path to the math_cots directory
        model_name: Name of the model
        
    Returns:
        List of solution file paths
    """
    # Convert model name to directory format (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" -> "deepseek-r1-distill-qwen-14b")
    model_dir_name = model_name.split('/')[-1].lower()
    
    # Find all solution files for this model
    solution_files = []
    for model_dir in math_cots_dir.iterdir():
        if model_dir.is_dir() and model_dir_name in model_dir.name.lower():
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and "temperature" in subdir.name:
                    solution_file = subdir / "solutions.json"
                    if solution_file.exists():
                        solution_files.append(solution_file)
    
    return solution_files

def compute_kl_attribution(
    thinking_model: AutoModelForCausalLM, 
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    thinking_text: str,
    target_module_name: str = "attention",
    layer_indices: List[int] = None
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Compute KL divergence attribution for a thinking text.
    
    Args:
        thinking_model: The thinking model
        base_model: The base model
        tokenizer: Tokenizer for both models
        thinking_text: Text with thinking process
        target_module_name: Name of the module to analyze (attention, mlp, or all)
        layer_indices: List of layer indices to analyze
        
    Returns:
        Tuple of attribution scores and tokens
    """
    device = next(thinking_model.parameters()).device
    
    # Tokenize the thinking text
    inputs = tokenizer(thinking_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)
    
    # When preparing tokens for visualization
    cleaned_tokens = []
    for token in tokens:
        # Remove the special space character
        if token.startswith('Ġ'):
            cleaned_tokens.append(token[1:])
        else:
            cleaned_tokens.append(token)
    
    # Register hooks to capture activations and gradients
    activations = {}
    gradients = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    def get_gradient(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0]
        return hook
    
    # Register hooks based on target module
    if layer_indices is None:
        layer_indices = list(range(thinking_model.config.num_hidden_layers))
    
    for layer_idx in layer_indices:
        if target_module_name == "attention" or target_module_name == "all":
            # Register hooks for attention module
            attn_module = thinking_model.model.layers[layer_idx].self_attn
            hooks.append(attn_module.register_forward_hook(get_activation(f"attn_{layer_idx}")))
            hooks.append(attn_module.register_full_backward_hook(get_gradient(f"attn_{layer_idx}")))
        
        if target_module_name == "mlp" or target_module_name == "all":
            # Register hooks for MLP module
            mlp_module = thinking_model.model.layers[layer_idx].mlp
            hooks.append(mlp_module.register_forward_hook(get_activation(f"mlp_{layer_idx}")))
            hooks.append(mlp_module.register_full_backward_hook(get_gradient(f"mlp_{layer_idx}")))
    
    # Get base model logits (no gradients)
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_outputs.logits
    
    # Get thinking model logits (with gradients)
    thinking_outputs = thinking_model(input_ids=input_ids, attention_mask=attention_mask)
    thinking_logits = thinking_outputs.logits
    
    # Compute KL divergence for each token position
    kl_divergences = []
    
    for pos in range(input_ids.shape[1] - 1):  # Exclude the last token
        # Get logits for the next token
        base_next_token_logits = base_logits[:, pos, :]
        thinking_next_token_logits = thinking_logits[:, pos, :]
        
        # Convert to probabilities
        base_probs = torch.nn.functional.softmax(base_next_token_logits, dim=-1)
        thinking_log_probs = torch.nn.functional.log_softmax(thinking_next_token_logits, dim=-1)
        
        # Compute KL divergence: base_probs * (log(base_probs) - thinking_log_probs)
        # We only want gradients through thinking_log_probs
        base_log_probs = torch.log(base_probs + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Detach base_probs and base_log_probs to prevent gradients
        kl_div = (base_probs.detach() * (base_log_probs.detach() - thinking_log_probs)).sum(dim=-1)
        kl_divergences.append(kl_div)
    
    # Sum KL divergences to get total loss
    total_kl = torch.stack(kl_divergences).sum()
    
    # Compute gradients
    total_kl.backward()
    
    # Compute attribution scores
    attribution_scores = {}
    
    for layer_idx in layer_indices:
        if target_module_name == "attention" or target_module_name == "all":
            # Compute attribution for attention module
            attn_key = f"attn_{layer_idx}"
            if attn_key in activations and attn_key in gradients:
                attribution_scores[attn_key] = (activations[attn_key][0] * gradients[attn_key][0]).sum(dim=2).abs()
        
        if target_module_name == "mlp" or target_module_name == "all":
            # Compute attribution for MLP module
            mlp_key = f"mlp_{layer_idx}"
            if mlp_key in activations and mlp_key in gradients:
                attribution_scores[mlp_key] = (activations[mlp_key] * gradients[mlp_key]).sum(dim=2).abs()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attribution_scores, cleaned_tokens

def analyze_solution_kl(
    solution: Dict,
    thinking_model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_module_name: str = "attention",
    layer_indices: List[int] = None,
    force: bool = False
) -> None:
    """
    Analyze a solution using KL divergence attribution.
    
    Args:
        solution: Solution dictionary
        thinking_model: The thinking model
        base_model: The base model
        tokenizer: Tokenizer for both models
        target_module_name: Name of the module to analyze (attention, mlp, or all)
        layer_indices: List of layer indices to analyze
        force: Whether to force recomputation of existing results
    """
    # Create output directory for this problem
    problem_idx = solution["problem_idx"]
    problem_output_dir = output_dir / f"problem_{problem_idx}"
    problem_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if attribution already exists
    attribution_file = problem_output_dir / f"attribution_{target_module_name}.pt"
    token_file = problem_output_dir / "tokens.json"
    
    if attribution_file.exists() and token_file.exists() and not force:
        print(f"Attribution for problem {problem_idx} already exists. Skipping...")
        return
    
    # Extract thinking text from the solution
    thinking_text = None
    if "<think>" in solution["full_cot"] and "</think>" in solution["full_cot"]:
        thinking_text = solution["full_cot"].split("<think>")[1].split("</think>")[0].strip() # NOTE: Is this right?
    else:
        thinking_text = solution.get("reasoning", "")
    
    if not thinking_text:
        print(f"No thinking text found for problem {problem_idx}. Skipping...")
        return
    
    print(f"Computing KL attribution for problem {problem_idx}...")
    
    try:
        # Compute KL attribution
        attribution_scores, tokens = compute_kl_attribution(
            thinking_model,
            base_model,
            tokenizer,
            thinking_text,
            target_module_name,
            layer_indices
        )
        
        # Save attribution scores and tokens
        torch.save(attribution_scores, attribution_file)
        
        with open(token_file, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
        
        # Visualize attribution scores
        visualize_attribution(attribution_scores, tokens, problem_output_dir, target_module_name)
        
        print(f"Attribution for problem {problem_idx} saved to {attribution_file}")
    
    except Exception as e:
        print(f"Error computing attribution for problem {problem_idx}: {e}")
        traceback.print_exc()
    
    # Add memory cleanup after each solution
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def visualize_attribution(
    attribution_scores: Dict[str, torch.Tensor],
    tokens: List[str],
    output_dir: Path,
    target_module_name: str
) -> None:
    """
    Visualize attribution scores.
    
    Args:
        attribution_scores: Dictionary of attribution scores
        tokens: List of tokens
        output_dir: Output directory
        target_module_name: Name of the module analyzed
    """
    # Create visualization for each component
    for component_name, scores in attribution_scores.items():
        # Convert to numpy for plotting
        if scores.dim() > 1:
            # If scores have multiple dimensions (e.g., for attention), take the mean
            scores_np = scores[0].detach().cpu().numpy()
        else:
            scores_np = scores.detach().cpu().numpy()
        
        # Ensure scores match token length
        if len(scores_np) > len(tokens):
            scores_np = scores_np[:len(tokens)]
        elif len(scores_np) < len(tokens):
            # Pad with zeros
            scores_np = np.pad(scores_np, (0, len(tokens) - len(scores_np)))
        
        # Create heatmap
        plt.figure(figsize=(20, 10))
        
        # Plot scores as a heatmap
        sns.heatmap(
            scores_np.reshape(1, -1),
            cmap="viridis",
            annot=False,
            fmt=".2f",
            cbar=True,
            xticklabels=tokens
        )
        
        plt.title(f"KL Attribution Scores - {component_name}")
        plt.xlabel("Tokens")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_dir / f"attribution_{component_name}.png")
        plt.close()
        
        # Create bar plot for top tokens
        plt.figure(figsize=(15, 10))
        
        # Get top 50 tokens by attribution score
        top_indices = np.argsort(scores_np)[-50:]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = scores_np[top_indices]
        
        # Plot as bar chart
        plt.bar(range(len(top_tokens)), top_scores)
        plt.xticks(range(len(top_tokens)), top_tokens, rotation=90)
        plt.xlabel("Tokens")
        plt.ylabel("Attribution Score")
        plt.title(f"Top 50 Tokens by KL Attribution - {component_name}")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"top_tokens_{component_name}.png")
        plt.close()

def analyze_common_patterns(output_dir: Path, target_module_name: str, top_k: int = 100) -> None:
    """
    Analyze common patterns across all problems.
    
    Args:
        output_dir: Output directory
        target_module_name: Name of the module analyzed
        top_k: Number of top tokens to consider
    """
    # Find all attribution files
    attribution_files = []
    for problem_dir in output_dir.iterdir():
        if problem_dir.is_dir() and problem_dir.name.startswith("problem_"):
            attribution_file = problem_dir / f"attribution_{target_module_name}.pt"
            if attribution_file.exists():
                attribution_files.append(attribution_file)
    
    if not attribution_files:
        print("No attribution files found. Skipping common patterns analysis.")
        return
    
    # Collect top tokens from each problem
    all_top_tokens = {}
    
    for attribution_file in attribution_files:
        # Find corresponding token file
        problem_dir = attribution_file.parent
        token_file = problem_dir / "tokens.json"
        
        if not token_file.exists():
            continue
        
        # Load tokens
        with open(token_file, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        # Load attribution scores
        attribution_scores = torch.load(attribution_file)
        
        # Process each component's attribution scores
        for component_name, scores in attribution_scores.items():
            # Convert to numpy for easier handling
            if scores.dim() > 1:
                # If scores have multiple dimensions (e.g., for attention), take the mean
                scores_np = scores.mean(dim=tuple(range(1, scores.dim()))).detach().cpu().numpy()
            else:
                scores_np = scores.detach().cpu().numpy()
            
            # Ensure scores match token length
            if len(scores_np) > len(tokens):
                scores_np = scores_np[:len(tokens)]
            elif len(scores_np) < len(tokens):
                # Pad with zeros
                scores_np = np.pad(scores_np, (0, len(tokens) - len(scores_np)))
            
            # Get top k tokens
            top_indices = np.argsort(scores_np)[-top_k:]
            
            # Add to collection
            component_key = f"{target_module_name}_{component_name}"
            if component_key not in all_top_tokens:
                all_top_tokens[component_key] = {}
            
            for idx in top_indices:
                if idx >= len(tokens):
                    continue
                    
                token = tokens[idx]
                score = scores_np[idx]
                
                if token not in all_top_tokens[component_key]:
                    all_top_tokens[component_key][token] = []
                
                all_top_tokens[component_key][token].append(score)
    
    # For each component, calculate average score for each token
    for component_key, token_scores in all_top_tokens.items():
        token_avg_scores = {}
        for token, scores in token_scores.items():
            token_avg_scores[token] = np.mean(scores)
        
        # Sort by average score
        sorted_tokens = sorted(token_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Save results
        with open(output_dir / f"common_patterns_{component_key}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Top {len(sorted_tokens)} tokens by average KL divergence attribution for {component_key}:\n\n")
            for token, score in sorted_tokens:
                f.write(f"{token}: {score:.6f} (appears in {len(token_scores[token])} problems)\n")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot top 50 tokens
        top_50_tokens = sorted_tokens[:50]
        tokens = [t[0] for t in top_50_tokens]
        scores = [t[1] for t in top_50_tokens]
        
        plt.bar(range(len(tokens)), scores)
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.xlabel("Token")
        plt.ylabel("Average Attribution Score")
        plt.title(f"Top 50 Tokens by Average KL Divergence Attribution - {component_key}")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"common_patterns_{component_key}.png")
        plt.close()
        
        print(f"Common patterns analysis for {component_key} saved to {output_dir / f'common_patterns_{component_key}.txt'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze CoTs using KL divergence attribution")
    parser.add_argument("--thinking_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Thinking model name (e.g., DeepSeek-R1-Distill-Qwen-14B)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-14B", help="Base model name (e.g., Qwen2.5-14B)")
    parser.add_argument("--layer", type=int, default=47, help="Layer to analyze")
    parser.add_argument("--module", type=str, default="attention", choices=["attention", "mlp", "all"], help="Module to analyze")
    parser.add_argument("--max_solutions", type=int, default=None, help="Maximum number of solutions to analyze")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing results")
    args = parser.parse_args()
    
    # Load thinking model and tokenizer
    thinking_model, tokenizer = load_model_and_tokenizer(args.thinking_model)
    
    # Load base model
    base_model, _ = load_model_and_tokenizer(args.base_model)
    
    # Find solution files for the thinking model
    solution_files = get_solution_files(math_cots_dir, args.thinking_model)
    
    if not solution_files:
        print(f"No solution files found for model {args.thinking_model}")
        return
    
    # Process each solution file
    for solution_file in solution_files:
        print(f"Processing solutions from {solution_file}")
        
        # Load solutions
        solutions = load_solutions(solution_file)
        
        # Limit number of solutions if specified
        if args.max_solutions:
            solutions = solutions[:args.max_solutions]
        
        # Analyze each solution
        for i, solution in enumerate(tqdm(solutions, desc="Analyzing solutions")):
            analyze_solution_kl(
                solution,
                thinking_model,
                base_model,
                tokenizer,
                target_module_name=args.module,
                layer_indices=[args.layer],
                force=args.force
            )
            
            # Add periodic more aggressive cleanup every 5 solutions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Print current memory usage for monitoring
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Analyze common patterns
    analyze_common_patterns(output_dir, args.module)

if __name__ == "__main__":
    main()