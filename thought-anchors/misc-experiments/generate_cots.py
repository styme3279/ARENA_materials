import os
import json
import random
import numpy as np
import torch
import asyncio
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from datasets import load_dataset
import argparse
from utils import extract_boxed_answers, check_answer
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Get OpenRouter API key
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Initialize tokenizer for logit bias
tokenizer = None

def initialize_tokenizer(model_name):
    """Initialize the tokenizer for the specified model."""
    global tokenizer
    try:
        # Extract the base model name from the OpenRouter model string
        if "/" in model_name:
            base_model = model_name.split("/")[-1]
        else:
            base_model = model_name
            
        # Map to Hugging Face model name
        hf_model_map = {
            "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "qwen-14b": "Qwen/Qwen-14B",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
        }
        
        # Get the Hugging Face model name
        hf_model = hf_model_map.get(base_model, f"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        
        print(f"Initializing tokenizer for {hf_model}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        return True
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        print("Using default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        return False

def load_math_problems(problem_type: Optional[str] = None, level: Optional[str] = None, num_problems: Optional[int] = None, split: str = 'train') -> List[Tuple[int, Dict]]:
    """
    Load problems from the MATH dataset with optional filtering.
    
    Args:
        problem_type: Type of problems to filter by (if None, use all types)
        level: Level of problems to filter by (if None, use all levels)
        num_problems: Number of problems to sample (if None, use all problems)
        split: Dataset split to use ('train' or 'test')
        
    Returns:
        List of problems with their original indices
    """
    try:
        # Load from Hugging Face dataset
        math_dataset = load_dataset("fdyrd/math")
        dataset_split = math_dataset[split]
        
        # Add original indices to problems
        indexed_problems = [(i, {
            'problem': item['problem'],
            'level': item['level'],
            'type': item['type'],
            'gt_solution': item['solution']
        }) for i, item in enumerate(dataset_split)]
        
        # Extract ground truth answers
        for i, problem in indexed_problems:
            gt_boxed_answers = extract_boxed_answers(problem['gt_solution'])
            gt_answer = gt_boxed_answers[0] if gt_boxed_answers else ""
            problem['gt_answer'] = gt_answer
        
        # Filter by type if specified
        if problem_type is not None:
            indexed_problems = [(i, problem) for i, problem in indexed_problems if problem.get('type') == problem_type]
        
        # Filter by level if specified
        if level is not None:
            indexed_problems = [(i, problem) for i, problem in indexed_problems if problem.get('level') == level]
            
        # Sample if needed
        if num_problems is not None and num_problems < len(indexed_problems):
            indexed_problems = random.sample(indexed_problems, num_problems)
            
        if level:
            print(f"Filtered to level: {level}")
        if problem_type:
            print(f"Filtered to type: {problem_type}")
            
        return indexed_problems
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []

async def make_openrouter_request(prompt: str, model: str, temperature: float, top_p: float, max_tokens: int, provider: str, logit_bias: Optional[Dict[str, float]] = None) -> Dict:
    """Make a direct HTTP request to OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "include_reasoning": True,
        "provider": {
            "order": [provider],
            "ignore": ["Together" if provider == "Novita" else "Novita"],
            "allow_fallbacks": False
        }
    }
    
    # Add logit bias if provided
    if logit_bias:
        payload["logit_bias"] = logit_bias
    
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        return response.json()

async def generate_solution(problem: Dict, model: str, temperature: float, top_p: float, max_tokens: int, provider: str, run_id: int = 0, logit_bias: Optional[Dict[str, float]] = None) -> Dict:
    """
    Generate a solution for a problem using OpenRouter API.
    
    Args:
        problem: Problem dictionary
        model: Model to use
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens for generation
        provider: Provider to use
        run_id: Run identifier for multiple runs of the same problem
        logit_bias: Logit bias dictionary
        
    Returns:
        Dictionary with the generated solution
    """
    # Create prompt
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = await make_openrouter_request(prompt, model, temperature, top_p, max_tokens, provider, logit_bias)
            
            solution_text = response['choices'][0]['message']['content']
            
            # Try to get reasoning tokens if available
            reasoning = None
            try:
                reasoning = response['choices'][0]['message']['reasoning']
            except (KeyError, TypeError):
                print("Reasoning tokens not available in response")
            
            # Create full CoT with prompt, reasoning, and solution
            full_cot = f"{prompt}{solution_text}"
            
            if reasoning:
                full_cot = f"{prompt}{reasoning}\n</think>\n{solution_text}"
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(solution_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "problem_idx": problem.get("problem_idx", -1),
                "problem": problem['problem'],
                "level": problem.get('level', "Unknown"),
                "type": problem.get('type', "Unknown"),
                "prompt": prompt,
                "solution": solution_text,
                "reasoning": reasoning,
                "full_cot": full_cot,
                "temperature": temperature,
                "top_p": top_p,
                "answer": answer,
                "gt_answer": problem.get('gt_answer', ""),
                "is_correct": is_correct,
                "run_id": run_id
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "problem_idx": problem.get("problem_idx", -1),
                    "problem": problem['problem'],
                    "level": problem.get('level', "Unknown"),
                    "type": problem.get('type', "Unknown"),
                    "prompt": prompt,
                    "solution": f"Error: {str(e)}",
                    "temperature": temperature,
                    "top_p": top_p,
                    "error": str(e),
                    "run_id": run_id
                }

async def generate_solutions_parallel(problems: List[Tuple[int, Dict]], args) -> List[Dict]:
    """
    Generate solutions for multiple problems in parallel.
    
    Args:
        problems: List of problems with their original indices
        args: Command-line arguments
        
    Returns:
        List of solution dictionaries
    """
    # Check if solutions file already exists
    output_dir = Path(args.output_dir) / args.model.split("/")[-1]
    if args.logit_bias_tokens and args.logit_bias_strength is not None:
        output_dir = output_dir / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}_logit_bias_{args.logit_bias_tokens}_{args.logit_bias_strength}"
    else:
        output_dir = output_dir / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
    solutions_file = output_dir / f"solutions.json"
    
    existing_solutions = []
    if solutions_file.exists():
        print(f"Loading existing solutions from {solutions_file}")
        with open(solutions_file, 'r', encoding='utf-8') as f:
            existing_solutions = json.load(f)
    
    # Create a dictionary of existing solutions by problem_idx and run_id
    existing_solutions_dict = {}
    solutions_to_retry = []
    
    # Recalculate accuracy for existing solutions
    if existing_solutions and not args.skip_recalculate:
        print(f"Recalculating accuracy for {len(existing_solutions)} existing solutions...")
        updated_count = 0
        
        for sol in existing_solutions:
            # Skip solutions with errors
            if 'error' in sol:
                solutions_to_retry.append((sol.get('problem_idx', -1), sol.get('run_id', 0)))
                continue
                
            # Extract answer and check correctness with updated check_answer function
            extracted_answers = extract_boxed_answers(sol['solution'])
            answer = extracted_answers[0] if extracted_answers else ""
            
            # Check if answer is empty
            if not answer:
                solutions_to_retry.append((sol.get('problem_idx', -1), sol.get('run_id', 0)))
                continue
                
            # Recalculate correctness
            is_correct = False
            if sol.get('gt_answer') and answer:
                is_correct = check_answer(answer, sol['gt_answer'])
            
            # Check if accuracy changed
            if sol.get('answer') != answer or sol.get('is_correct') != is_correct:
                updated_count += 1
            
            # Update the accuracy fields directly in the solution
            sol['answer'] = answer
            sol['is_correct'] = is_correct
            
            # Add to dictionary of valid solutions
            problem_idx = sol.get('problem_idx', -1)
            run_id = sol.get('run_id', 0)
            
            if problem_idx not in existing_solutions_dict:
                existing_solutions_dict[problem_idx] = {}
            existing_solutions_dict[problem_idx][run_id] = sol
        
        print(f"Updated accuracy for {updated_count} solutions")
    else:
        # If not recalculating, just organize solutions and identify retries
        for sol in existing_solutions:
            problem_idx = sol.get('problem_idx', -1)
            run_id = sol.get('run_id', 0)
            
            # Check if this solution had an error or empty answer
            has_error = 'error' in sol
            has_empty_answer = not sol.get('answer', '')
            
            if has_error or has_empty_answer:
                # Mark this solution for retry
                solutions_to_retry.append((problem_idx, run_id))
                continue
                
            # Add to dictionary of valid solutions
            if problem_idx not in existing_solutions_dict:
                existing_solutions_dict[problem_idx] = {}
            existing_solutions_dict[problem_idx][run_id] = sol
    
    if solutions_to_retry:
        print(f"Found {len(solutions_to_retry)} solutions with errors or empty answers to retry")
    
    # Create tasks for problems/runs that don't already exist or need to be retried
    tasks = []
    for i, (problem_idx, problem) in enumerate(problems):
        # Add problem index to the problem dictionary
        problem['problem_idx'] = problem_idx
        
        # Create multiple tasks for each problem if runs_per_problem > 1
        for run_id in range(args.runs_per_problem):
            # Skip if this problem/run combination already exists and doesn't need retry
            if (problem_idx in existing_solutions_dict and 
                run_id in existing_solutions_dict[problem_idx] and
                (problem_idx, run_id) not in solutions_to_retry):
                continue
                
            # Initialize tokenizer if logit bias is requested
            logit_bias = None
            if args.logit_bias_tokens and args.logit_bias_strength is not None:
                if tokenizer is None:
                    initialize_tokenizer(args.model)
                logit_bias = create_logit_bias(args.logit_bias_tokens, args.logit_bias_strength)
            
            task = generate_solution(
                problem=problem,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                provider=args.provider,
                run_id=run_id,
                logit_bias=logit_bias
            )
            tasks.append(task)
    
    # If no new tasks, return existing solutions
    if not tasks:
        print("All solutions already exist and are valid. No new computations needed.")
        # Rebuild the list of solutions from the dictionary to ensure we have the updated versions
        updated_solutions = []
        for problem_dict in existing_solutions_dict.values():
            updated_solutions.extend(problem_dict.values())
        return updated_solutions
    
    # Execute tasks with limited concurrency
    print(f"Generating {len(tasks)} new solutions with max concurrency of {args.concurrency}...")
    new_solutions = []
    
    # Process tasks in batches to limit concurrency
    for i in range(0, len(tasks), args.concurrency):
        batch = tasks[i:i + args.concurrency]
        for future in tqdm(asyncio.as_completed(batch), total=len(batch)):
            solution = await future
            new_solutions.append(solution)
    
    # Combine existing and new solutions
    # First rebuild the list of existing solutions from the dictionary to ensure we have the updated versions
    updated_existing_solutions = []
    for problem_dict in existing_solutions_dict.values():
        updated_existing_solutions.extend(problem_dict.values())
    
    all_solutions = updated_existing_solutions + new_solutions
    
    return all_solutions

def analyze_results(solutions: List[Dict]) -> Dict:
    """
    Analyze the results of the solutions.
    
    Args:
        solutions: List of solution dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    # Group solutions by problem_idx
    problem_results = {}
    for sol in solutions:
        problem_idx = sol.get('problem_idx', -1)
        if problem_idx not in problem_results:
            problem_results[problem_idx] = []
        problem_results[problem_idx].append(sol)
    
    # Calculate metrics for each problem
    problem_metrics = {}
    for problem_idx, sols in problem_results.items():
        total_runs = len(sols)
        correct = sum(1 for sol in sols if sol.get('is_correct', False))
        incorrect = sum(1 for sol in sols if not sol.get('is_correct', False) and sol.get('answer', '') and 'error' not in sol)
        empty = sum(1 for sol in sols if not sol.get('answer', ''))
        errors = sum(1 for sol in sols if 'error' in sol)
        
        accuracy = correct / total_runs if total_runs > 0 else 0
        
        # Get problem metadata from first solution
        first_sol = sols[0]
        
        problem_metrics[problem_idx] = {
            'problem': first_sol.get('problem', ''),
            'level': first_sol.get('level', 'Unknown'),
            'type': first_sol.get('type', 'Unknown'),
            'gt_answer': first_sol.get('gt_answer', ''),
            'total_runs': total_runs,
            'correct': correct,
            'incorrect': incorrect,
            'empty': empty,
            'errors': errors,
            'accuracy': accuracy,
            'solutions': sols
        }
    
    # Calculate overall metrics
    total_runs = len(solutions)
    correct = sum(1 for sol in solutions if sol.get('is_correct', False))
    incorrect = sum(1 for sol in solutions if not sol.get('is_correct', False) and sol.get('answer', '') and 'error' not in sol)
    empty = sum(1 for sol in solutions if not sol.get('answer', ''))
    errors = sum(1 for sol in solutions if 'error' in sol)
    
    overall_accuracy = correct / total_runs if total_runs > 0 else 0
    
    # Group by level
    level_results = {}
    for sol in solutions:
        level = sol.get('level', 'Unknown')
        if level not in level_results:
            level_results[level] = {'total': 0, 'correct': 0, 'incorrect': 0, 'empty': 0, 'errors': 0}
        
        level_results[level]['total'] += 1
        if sol.get('is_correct', False):
            level_results[level]['correct'] += 1
        elif not sol.get('answer', '') or sol.get('answer', '') == '':
            level_results[level]['empty'] += 1
        elif 'error' in sol:
            level_results[level]['errors'] += 1
        else:
            level_results[level]['incorrect'] += 1
    
    # Group by type
    type_results = {}
    for sol in solutions:
        problem_type = sol.get('type', 'Unknown')
        if problem_type not in type_results:
            type_results[problem_type] = {'total': 0, 'correct': 0, 'incorrect': 0, 'empty': 0, 'errors': 0}
        
        type_results[problem_type]['total'] += 1
        if sol.get('is_correct', False):
            type_results[problem_type]['correct'] += 1
        elif not sol.get('answer', '') or sol.get('answer', '') == '':
            type_results[problem_type]['empty'] += 1
        elif 'error' in sol:
            type_results[problem_type]['errors'] += 1
        else:
            type_results[problem_type]['incorrect'] += 1
    
    return {
        'overall': {
            'total_runs': total_runs,
            'correct': correct,
            'incorrect': incorrect,
            'empty': empty,
            'errors': errors,
            'accuracy': overall_accuracy
        },
        'by_problem': problem_metrics,
        'by_level': level_results,
        'by_type': type_results
    }

def create_logit_bias(tokens_str: str, bias_strength: int) -> Dict[str, float]:
    """
    Create a logit bias dictionary from a comma-separated string of tokens.
    
    Args:
        tokens_str: Comma-separated string of tokens
        bias_strength: Strength of the bias (-100 to 100)
        
    Returns:
        Dictionary mapping token IDs to bias values
    """
    if not tokens_str or not tokenizer:
        return {}
        
    # Clamp bias strength to valid range
    bias_strength = max(-100, min(100, bias_strength))
    
    # Split tokens and create bias dictionary
    logit_bias = {}
    tokens = [token.strip() for token in tokens_str.split(',')]
    tokens_extended = []
    for token in tokens:
        tokens_extended.append(token.lower())
        tokens_extended.append(token[0].upper() + token[1:].lower())
        tokens_extended.append(f" {token.lower()}")
        tokens_extended.append(f" {token[0].upper() + token[1:].lower()}")
    
    for token in tokens_extended:
        if not token:
            continue
            
        # Get token ID
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        
        for token_id in token_ids:
            logit_bias[str(token_id)] = bias_strength
            
    return logit_bias

async def main():
    parser = argparse.ArgumentParser(description='Generate solutions for math problems')
    parser.add_argument('-m', '--model', type=str, default="deepseek/deepseek-r1-distill-qwen-14b", help='Model to use')
    parser.add_argument('-o', '--output_dir', type=str, default='math_cots', help='Directory to save results')
    parser.add_argument('-np', '--num_problems', type=int, default=1000, help='Number of problems to sample')
    parser.add_argument('-r', '--runs_per_problem', type=int, default=10, help='Number of runs per problem')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('-tp', '--top_p', type=float, default=0.92, help='Top-p sampling parameter')
    parser.add_argument('-mt', '--max_tokens', type=int, default=16384, help='Maximum number of tokens for generation')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
    parser.add_argument('-l', '--level', type=str, default="Level 5", help='Problem level filter')
    parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('-p', '--provider', type=str, default="Novita", choices=['Novita', 'Together'], help='Provider to use')
    parser.add_argument('-lbt', '--logit_bias_tokens', type=str, default=None, help='Comma-separated tokens to apply logit bias to')
    parser.add_argument('-lbs', '--logit_bias_strength', type=int, default=None, help='Strength of logit bias (-100 to 100)')
    parser.add_argument('-c', '--concurrency', type=int, default=200, help='Maximum number of concurrent requests')
    parser.add_argument('-sr', '--skip_recalculate', default=False, action='store_true', help='Skip recalculating accuracy for existing solutions')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model.split("/")[-1]
    if args.logit_bias_tokens and args.logit_bias_strength is not None:
        output_dir = output_dir / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}_logit_bias_{args.logit_bias_tokens}_{args.logit_bias_strength}"
    else:
        output_dir = output_dir / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load problems
    problems = load_math_problems(problem_type=args.type, level=args.level, num_problems=args.num_problems, split=args.split)
    
    if not problems:
        print("No problems found with the specified criteria.")
        return
    
    # Generate solutions
    solutions = await generate_solutions_parallel(problems, args)
    
    # Save all solutions
    solutions_file = output_dir / f"solutions.json"
    with open(solutions_file, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, indent=2)
    
    # Analyze results
    results = analyze_results(solutions)
    
    # Add logit bias information to results
    if args.logit_bias_tokens and args.logit_bias_strength is not None:
        results['logit_bias'] = {'tokens': args.logit_bias_tokens, 'strength': args.logit_bias_strength}
    
    # Save detailed results
    results_file = output_dir / f"results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print overall accuracy
    overall = results['overall']
    print(f"\n===== Overall Results =====")
    print(f"Total runs: {overall['total_runs']}")
    print(f"Overall accuracy: {overall['accuracy']*100:.2f}%")
    
    # Find and print the hardest problems (lowest accuracy)
    problem_metrics = results['by_problem']
    hardest_problems = sorted(problem_metrics.items(), key=lambda x: x[1]['accuracy'])
    
    # Save hardest problems to a separate file
    hardest_problems_data = {idx: metrics for idx, metrics in hardest_problems}
    hardest_file = output_dir / f"hardest_problems.json"
    with open(hardest_file, 'w', encoding='utf-8') as f:
        json.dump(hardest_problems_data, f, indent=2)
    
    # Print the 10 hardest problems
    print("\n===== Hardest Problems =====")
    for i, (problem_idx, metrics) in enumerate(hardest_problems[:10]):
        print(f"\n{i+1}. Problem {problem_idx} ({metrics['level']}, {metrics['type']})")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['total_runs']} correct)")
        print(f"   Problem: {metrics['problem']}")
        print(f"   Ground truth: {metrics['gt_answer']}")
    
    print(f"\nResults saved to {results_file}")
    print(f"Solutions saved to {solutions_file}")
    print(f"Hardest problems saved to {hardest_file}")

    # Print logit bias info if used
    if args.logit_bias_tokens and args.logit_bias_strength is not None:
        print(f"\nLogit bias applied to tokens: {args.logit_bias_tokens}")
        print(f"Logit bias strength: {args.logit_bias_strength}")

if __name__ == "__main__":
    asyncio.run(main())