import os
import json
import random
import numpy as np
import torch
import asyncio
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from utils import extract_boxed_answers, check_answer, split_solution_into_chunks, load_math_problems
from transformers import TextStreamer

# Load environment variables
load_dotenv()

# Get API keys
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Set up argument parser
import argparse
parser = argparse.ArgumentParser(description='Generate chain-of-thought solutions with rollouts')
parser.add_argument('-m', '--model', type=str, default="deepseek/deepseek-r1-distill-qwen-14b", help='Model to use') # "deepseek/deepseek-r1-distill-llama-8b"
parser.add_argument('-b', '--base_solution_type', type=str, default='correct', choices=['correct', 'incorrect'], help='Type of base solution to generate')
parser.add_argument('-r', '--rollout_type', type=str, default='default', choices=['default', 'forced_answer'], help='Type of rollout to generate')
parser.add_argument('-o', '--output_dir', type=str, default='math_rollouts', help='Directory to save results')
parser.add_argument('-np', '--num_problems', type=int, default=100, help='Number of problems to sample')
parser.add_argument('-nr', '--num_rollouts', type=int, default=100, help='Number of rollouts per chunk')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.95, help='Top-p sampling parameter')
parser.add_argument('-mt', '--max_tokens', type=int, default=16384, help='Maximum number of tokens for generation')
parser.add_argument('-mc', '--max_chunks', type=int, default=275, help='Maximum number of chunks to process')
parser.add_argument('-s', '--seed', type=int, default=44, help='Random seed for reproducibility')
parser.add_argument('-f', '--force', action='store_true', help='Force regeneration even if solutions exist')
parser.add_argument('-ep', '--exclude_problems', type=str, default=None, help='Comma-separated list of problem IDs to exclude')
parser.add_argument('-ip', '--include_problems', type=str, default=None, help='Comma-separated list of problem IDs to include')
parser.add_argument('-ic', '--include_chunks', type=str, default=None, help='Comma-separated list of chunk IDs to include')
parser.add_argument('-ty', '--type', type=str, default=None, help='Problem type filter')
parser.add_argument('-l', '--level', type=str, default="Level 5", help='Problem level filter')
parser.add_argument('-sp', '--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to use')
parser.add_argument('-p', '--provider', type=str, default="Novita", choices=['Novita', 'Together', 'Fireworks', 'Local'], help='Provider to use') # "Together"
parser.add_argument('-or', '--use_openrouter', default=False, action='store_true', help='Use OpenRouter API')
parser.add_argument('-fp', '--frequency_penalty', type=float, default=None, help='Frequency penalty parameter')
parser.add_argument('-pp', '--presence_penalty', type=float, default=None, help='Presence penalty parameter')
parser.add_argument('-rp', '--repetition_penalty', type=float, default=None, help='Repetition penalty parameter')
parser.add_argument('-tk', '--top_k', type=int, default=None, help='Top-k parameter')
parser.add_argument('-mp', '--min_p', type=float, default=None, help='Min-p parameter')
parser.add_argument('-sr', '--skip_recalculate', default=False, action='store_true', help='Skip recalculating accuracy for existing rollouts')
parser.add_argument('-q', '--quantize', default=False, action='store_true', help='Use quantization for local model')
parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for local model')
parser.add_argument('-mr', '--max_retries', type=int, default=1, help='Maximum number of retries for API requests')
parser.add_argument('-os', '--output_suffix', type=str, default=None, help='Suffix to add to the output directory')
args = parser.parse_args()

# Create output directory
base_output_dir = Path(args.output_dir) / args.model.split("/")[-1] / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
if args.rollout_type == 'forced_answer':
    # NOTE: For forced answer rollouts, we use the correct base solution (we copy the files from the correct base solution directory before running this script)
    output_dir = base_output_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}_{args.output_suffix}" if args.output_suffix else base_output_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}"
else:
    output_dir = base_output_dir / f"{args.base_solution_type}_base_solution_{args.output_suffix}" if args.output_suffix else base_output_dir / f"{args.base_solution_type}_base_solution"
output_dir.mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Load local model if using Local provider
local_model = None
local_tokenizer = None

if args.provider == "Local":
    try:
        print(f"Loading local model: {args.model}")
        model = args.model.replace("deepseek/", "deepseek-ai/") # Slight adjustment we need to make
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        local_tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Load model with quantization if specified
        if args.quantize and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            local_model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
            )
        else:
            local_model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            )
        
        print("Local model loaded successfully")
        local_model.eval()
    except Exception as e:
        print(f"Error loading local model: {e}")
        exit(1)

def generate_with_local_model(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Generate text using a local model."""
    try:
        # Tokenize the prompt
        inputs = local_tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Create a streamer that shows progress
        streamer = TextStreamer(local_tokenizer, skip_special_tokens=True)
        
        # Set up generation parameters
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "use_cache": True,
            "pad_token_id": local_tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        # Add optional parameters
        if args.top_k is not None:
            generation_config["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            generation_config["repetition_penalty"] = args.repetition_penalty
        
        # Generate
        with torch.no_grad():
            outputs = local_model.generate(**inputs, **generation_config)
        
        # Decode the generated text
        generated_text = local_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "finish_reason": "stop",  # Simplified
            "usage": {"total_tokens": len(outputs[0])}
        }
    except Exception as e:
        print(f"Error in local generation: {e}")
        return {"error": str(e)}

def generate_with_local_model_batch(prompts: List[str], temperature: float, top_p: float, max_tokens: int) -> List[Dict]:
    """Generate text using a local model in batch mode for multiple prompts."""
    try:
        results = []
        batch_size = args.batch_size
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Tokenize all prompts in the batch
            batch_inputs = local_tokenizer(batch_prompts, padding=True, return_tensors="pt")
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}
            
            # Set up generation parameters
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "use_cache": True,
                "pad_token_id": local_tokenizer.eos_token_id
            }
            
            # Add optional parameters
            if args.top_k is not None:
                generation_config["top_k"] = args.top_k
            if args.repetition_penalty is not None:
                generation_config["repetition_penalty"] = args.repetition_penalty
            
            # Generate
            with torch.no_grad():
                batch_outputs = local_model.generate(**batch_inputs, **generation_config)
            
            # Process each output in the batch
            for j, (input_ids, output_ids) in enumerate(zip(batch_inputs["input_ids"], batch_outputs)):
                # Find where the generated text starts (after the prompt)
                input_length = len(input_ids)
                
                # Decode the generated text
                generated_text = local_tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)
                
                results.append({
                    "text": generated_text,
                    "finish_reason": "stop",  # Simplified
                    "usage": {"total_tokens": len(output_ids)}
                })
        
        return results
    except Exception as e:
        print(f"Error in batch generation: {e}")
        return [{"error": str(e)} for _ in range(len(prompts))]

async def make_api_request(prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Make an API request to either Novita, Together, Fireworks, or use a local model based on provider setting."""
    # If using local model, use synchronous generation
    if args.provider == "Local":
        return generate_with_local_model(prompt, temperature, top_p, max_tokens)
    
    # Otherwise, use API-based generation
    if args.provider == "Novita":
        # Novita API request
        headers = {
            "Authorization": f"Bearer {NOVITA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": args.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False
        }
        
        api_url = "https://api.novita.ai/v3/openai/completions"
        
    elif args.provider == "Together":
        # Together API request
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        payload = {
            "model": "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        api_url = "https://api.together.xyz/v1/completions"
        
    elif args.provider == "Fireworks":
        # Fireworks API request
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "accounts/fireworks/models/deepseek-r1-distill-qwen-14b",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "stream": True
        }
        
        api_url = "https://api.fireworks.ai/inference/v1/completions"
    
    # Add optional parameters for all APIs
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None and args.provider != "Fireworks":  # Fireworks doesn't support min_p
        payload["min_p"] = args.min_p
    if args.seed is not None and False: # NOTE: We don't use seeds for rollouts
        payload["seed"] = args.seed
    
    # Implement exponential backoff for retries
    max_retries = args.max_retries
    retry_delay = 2 if max_retries > 0 else None
    
    for attempt in range(max_retries):
        try:
            # Handle streaming responses for Together and Fireworks
            if (args.provider == "Together" or args.provider == "Fireworks") and payload.get("stream", False):
                return await handle_streaming_response(api_url, headers, payload)
            
            # For non-streaming responses
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, headers=headers, json=payload, timeout=240)
                
                # Handle different error codes
                if response.status_code == 500:
                    print(f"Server error (500) on attempt {attempt+1}/{max_retries}. Retrying...")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                    
                elif response.status_code == 429:
                    print(f"Rate limit (429) on attempt {attempt+1}/{max_retries}. Retrying...")
                    await asyncio.sleep(retry_delay * (2 ** attempt) + random.uniform(1, 3))  # Add jitter
                    continue
                    
                elif response.status_code != 200:
                    print(f"Error from API: {response.status_code} - {response.text}")
                    
                    # If it's the last attempt, return the error
                    if attempt == max_retries - 1:
                        return {"error": f"API error: {response.status_code}", "details": response.text}
                    
                    # Otherwise retry
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                
                # Success case
                result = response.json()
                
                if args.provider == "Novita" or args.provider == "Together":
                    return {
                        "text": result["choices"][0]["text"],
                        "finish_reason": result["choices"][0].get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                elif args.provider == "Fireworks":
                    return {
                        "text": result["choices"][0]["text"],
                        "finish_reason": result["choices"][0].get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                
        except Exception as e:
            print(f"Exception during API request (attempt {attempt+1}/{max_retries}): {e}")
            
            # If it's the last attempt, return the error
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            
            # Otherwise retry
            await asyncio.sleep(retry_delay * (2 ** attempt))
    
    # If we get here, all retries failed
    return {"error": "All API request attempts failed"}

async def handle_streaming_response(api_url: str, headers: Dict, payload: Dict) -> Dict:
    """Handle streaming responses from Together or Fireworks API."""
    try:
        # Initialize variables to collect the response
        collected_text = ""
        finish_reason = None
        usage = None
        
        # Make the streaming request
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, headers=headers, json=payload, timeout=240) as response:
                if response.status_code != 200:
                    return {"error": f"API error: {response.status_code}", "details": await response.aread()}
                
                # Process the streaming response
                async for chunk in response.aiter_lines():
                    # Skip empty lines
                    if not chunk.strip():
                        continue
                    
                    # Check for the end of the stream
                    if chunk == "data: [DONE]":
                        break
                    
                    # Parse the chunk
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])  # Remove "data: " prefix
                            
                            # Extract text from the chunk
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                
                                # Get text content
                                if "text" in choice and choice["text"]:
                                    collected_text += choice["text"]
                                elif "delta" in choice and "content" in choice["delta"]:
                                    collected_text += choice["delta"]["content"]
                                
                                # Check for finish reason
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            
                            # Get usage information from the last chunk
                            if "usage" in data and data["usage"]:
                                usage = data["usage"]
                                
                        except json.JSONDecodeError:
                            print(f"Failed to parse chunk: {chunk}")
        
        # For Together API, we need to handle the <think> token and the newline after it
        if args.provider == "Together":
            if collected_text.startswith("<think>\n"):
                # Remove the <think> token and the newline after it
                collected_text = collected_text[len("<think>\n"):]
            elif collected_text.startswith("<think>"):
                # Remove just the <think> token if there's no newline
                collected_text = collected_text[len("<think>"):]
        
        return {
            "text": collected_text,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {}
        }
        
    except Exception as e:
        print(f"Exception during streaming: {e}")
        return {"error": f"Streaming exception: {str(e)}"}

async def generate_base_solution(problem: Dict, temperature: float = 0.6) -> Dict:
    """
    Generate a base solution for a problem using Novita API.
    
    Args:
        problem: Problem dictionary
        temperature: Temperature for generation
        
    Returns:
        Dictionary with the generated solution
    """
    # Create prompt similar to generate_cots_math.py
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n"
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            solution_text = response['text']
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(solution_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "prompt": prompt,
                "solution": solution_text,
                "full_cot": prompt + solution_text,
                "answer": answer,
                "is_correct": is_correct
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "prompt": prompt,
                    "solution": f"Error: {str(e)}",
                    "error": str(e)
                }

async def generate_rollout(problem: Dict, chunk_text: str, full_cot_prefix: str, temperature: float = 0.7, rollout_type: str = 'default') -> Dict:
    """
    Generate a rollout by removing a specific chunk and regenerating from that point.
    
    Args:
        problem: Problem dictionary
        chunk_text: Text of the current chunk to remove
        full_cot_prefix: Full CoT text up to and including the current chunk
        temperature: Temperature for generation
        rollout_type: Type of rollout to generate
    Returns:
        Dictionary with the rollout result
    """
    # Remove the current chunk from the prefix to see how it gets regenerated
    prefix_without_chunk = full_cot_prefix.replace(chunk_text, "").strip()
    
    # Create prompt with the prefix without the current chunk
    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
    
    if rollout_type == 'forced_answer':
        prompt += "\n</think>\n\nTherefore, the final answers is \\boxed{"
    
    max_retries = args.max_retries
    retry_delay = 2 if max_retries > 0 else None
    
    for attempt in range(max_retries):
        try:
            response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
            rollout_text = response['text']
            chunk_resampled = split_solution_into_chunks(rollout_text)[0]
            
            # Extract answer and check correctness
            extracted_answers = extract_boxed_answers(f"{prompt}{rollout_text}" if rollout_type == 'forced_answer' else rollout_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = False
            
            if problem.get('gt_answer') and answer:
                is_correct = check_answer(answer, problem['gt_answer'])
            
            return {
                "chunk_removed": chunk_text,
                "prefix_without_chunk": prefix_without_chunk,
                "chunk_resampled": chunk_resampled,
                "rollout": rollout_text,
                "full_cot": f"{prompt}{rollout_text}",
                "answer": answer,
                "is_correct": is_correct
            }
        except Exception as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "chunk_removed": chunk_text,
                    "prefix_without_chunk": prefix_without_chunk,
                    "error": str(e)
                }

async def process_problem(problem_idx: int, problem: Dict) -> None:
    """
    Process a single problem: generate base solution and rollouts.
    
    Args:
        problem_idx: Index of the problem
        problem: Problem dictionary
    """
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)
    
    # Save problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem, f, indent=2)
    
    # Check if base solution already exists
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, 'r', encoding='utf-8') as f:
            base_solution = json.load(f)
            print(f"Problem {problem_idx}: Loaded existing base solution")
            
            # Recalculate accuracy for base solution if needed
            if not args.skip_recalculate and 'solution' in base_solution:
                extracted_answers = extract_boxed_answers(base_solution['solution'])
                answer = extracted_answers[0] if extracted_answers else ""
                is_correct = False
                
                if problem.get('gt_answer') and answer:
                    is_correct = check_answer(answer, problem['gt_answer'])
                
                # Update if different
                if base_solution.get('answer') != answer or base_solution.get('is_correct') != is_correct:
                    print(f"Problem {problem_idx}: Updating base solution accuracy")
                    base_solution['answer'] = answer
                    base_solution['is_correct'] = is_correct
                    
                    # Save updated base solution
                    with open(base_solution_file, 'w', encoding='utf-8') as f:
                        json.dump(base_solution, f, indent=2)
    
    # Generate base solution if needed
    if base_solution is None:
        print(f"Problem {problem_idx}: Generating {args.base_solution_type} base solution")
        base_solution = await generate_base_solution(problem, args.temperature)
            
        if args.base_solution_type == "correct" and ("is_correct" not in base_solution or not base_solution["is_correct"]):
            print(base_solution["solution"])
            print(f"Problem {problem_idx}: Base solution is INCORRECT or has error. Retrying...")
            return await process_problem(problem_idx, problem)
        elif args.base_solution_type == "incorrect" and ("is_correct" not in base_solution or base_solution["is_correct"]):
            print(base_solution["solution"])
            print(f"Problem {problem_idx}: Base solution is CORRECT or has error. Retrying...")
            return await process_problem(problem_idx, problem)
        
        # Save base solution
        with open(base_solution_file, 'w', encoding='utf-8') as f:
            json.dump(base_solution, f, indent=2)
    
    # Get the source text for chunking
    source_text = base_solution["full_cot"]
    print(f"Problem {problem_idx}: Using full CoT for chunking")
    
    # Extract the solution part for chunking
    if "<think>" in source_text:
        solution_text = source_text.split("<think>")[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>")[0].strip()
    else:
        solution_text = source_text
    
    # Save chunks to a separate file
    chunks_file = problem_dir / "chunks.json"
    
    if not chunks_file.exists() or args.force:
        chunks = split_solution_into_chunks(solution_text)
        print(f"Problem {problem_idx}: Split into {len(chunks)} chunks")
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({"source_text": source_text, "solution_text": solution_text, "chunks": chunks}, f, indent=2)
        
        print(f"Problem {problem_idx}: Saved chunks to {chunks_file}")
    else:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)['chunks']
        print(f"Problem {problem_idx}: Loaded {len(chunks)} existing chunks")
        
    if len(chunks) > args.max_chunks:
        print(f"Problem {problem_idx}: Too many chunks. Will not generate rollouts.")
        return
    
    # Build cumulative chunks for proper continuation
    cumulative_chunks = []
    current_cumulative = ""
    for chunk in chunks:
        current_cumulative += chunk + " "
        cumulative_chunks.append(current_cumulative.strip())
    
    # Process each chunk
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        if args.include_chunks and str(chunk_idx) not in args.include_chunks.split(","):
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Skipping (not in include_chunks)")
            continue
        
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if solutions already exist
        solutions_file = chunk_dir / "solutions.json"
        existing_solutions = []
        valid_existing_solutions = []
        
        if solutions_file.exists() and not args.force:
            with open(solutions_file, 'r', encoding='utf-8') as f:
                existing_solutions = json.load(f)
                
                # Recalculate accuracy for existing rollouts if needed
                if not args.skip_recalculate:
                    updated_count = 0
                    for rollout in existing_solutions:
                        if 'rollout' in rollout and 'error' not in rollout:
                            extracted_answers = extract_boxed_answers(rollout['full_cot'] if args.rollout_type == 'forced_answer' else rollout['rollout'])
                            answer = extracted_answers[0] if extracted_answers else ""
                            is_correct = False
                            
                            if problem.get('gt_answer') and answer:
                                is_correct = check_answer(answer, problem['gt_answer'])
                            
                            # Update if different
                            if rollout.get('answer') != answer or rollout.get('is_correct') != is_correct:
                                updated_count += 1
                                rollout['answer'] = answer
                                rollout['is_correct'] = is_correct
                    
                    if updated_count > 0:
                        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Updated accuracy for {updated_count} rollouts")
                        # Save updated rollouts
                        with open(solutions_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_solutions, f, indent=2)
                
                # Filter for valid solutions (has answer and no error)
                valid_existing_solutions = [s for s in existing_solutions if 'answer' in s and 'error' not in s]
                print(f"Problem {problem_idx}, Chunk {chunk_idx}: Found {len(valid_existing_solutions)} valid solutions")
        
        # Generate rollouts if needed
        num_rollouts_needed = args.num_rollouts - len(valid_existing_solutions)
        
        if num_rollouts_needed > 0:
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {num_rollouts_needed} rollouts")
            
            # For Local provider, we can use batch processing
            if args.provider == "Local":
                # Create prompts for all rollouts
                prompts = []
                for _ in tqdm(range(num_rollouts_needed), desc="Generating rollouts"):
                    # Remove the current chunk from the prefix to see how it gets regenerated
                    prefix_without_chunk = full_prefix.replace(chunk, "").strip()
                    
                    # Create prompt with the prefix without the current chunk
                    prompt = f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
                    
                    if args.rollout_type == 'forced_answer':
                        prompt += "\n</think>\n\nTherefore, the final answers is \\boxed{"
                    
                    prompts.append(prompt)
                
                # Generate all rollouts in batch
                batch_results = generate_with_local_model_batch(prompts, args.temperature, args.top_p, args.max_tokens)
                
                # Process results
                new_solutions = []
                for i, result in enumerate(batch_results):
                    rollout_text = result.get('text', '')
                    
                    # Skip if there was an error
                    if 'error' in result:
                        new_solutions.append({"error": result['error']})
                        continue
                    
                    # Create the rollout object
                    prefix_without_chunk = full_prefix.replace(chunk, "").strip()
                    chunk_resampled = split_solution_into_chunks(rollout_text)[0] if rollout_text else ""
                    
                    # Extract answer and check correctness
                    prompt = prompts[i]
                    extracted_answers = extract_boxed_answers(f"{prompt}{rollout_text}" if args.rollout_type == 'forced_answer' else rollout_text)
                    answer = extracted_answers[0] if extracted_answers else ""
                    is_correct = False
                    
                    if problem.get('gt_answer') and answer:
                        is_correct = check_answer(answer, problem['gt_answer'])
                    
                    new_solutions.append({
                        "chunk_removed": chunk,
                        "prefix_without_chunk": prefix_without_chunk,
                        "chunk_resampled": chunk_resampled,
                        "rollout": rollout_text,
                        "full_cot": f"{prompt}{rollout_text}",
                        "answer": answer,
                        "is_correct": is_correct
                    })
            else:
                # For API providers, we can generate in parallel
                tasks = [generate_rollout(problem, chunk, full_prefix, args.temperature, args.rollout_type) for _ in range(num_rollouts_needed)]
                new_solutions = await asyncio.gather(*tasks)
            
            # Combine with existing solutions
            all_solutions = existing_solutions + new_solutions
            
            # Save all solutions
            with open(solutions_file, 'w', encoding='utf-8') as f:
                json.dump(all_solutions, f, indent=2)
            
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Saved {len(all_solutions)} solutions")
        else:
            print(f"Problem {problem_idx}, Chunk {chunk_idx}: Already have {len(valid_existing_solutions)} valid solutions")

async def main():
    """Main function to run the script."""
    # Load problems
    problems = load_math_problems(problem_type=args.type, level=args.level, num_problems=args.num_problems, split=args.split, include_problems=args.include_problems)
    
    if args.exclude_problems:
        exclude_problems = [int(id) for id in args.exclude_problems.split(",")]
        problems = [problem for problem in problems if problem[0] not in exclude_problems]
        
    if args.include_problems:
        include_problems = [int(id) for id in args.include_problems.split(",")]
        problems = [problem for problem in problems if problem[0] in include_problems]
    
    if not problems:
        print(f"No problems loaded. Exiting.")
        exit(1)

    print(f"Loaded {len(problems)} problems.")
    
    # Process problems
    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem)

if __name__ == "__main__":
    asyncio.run(main())

# Add this near the top of your script where you check for API keys
if args.provider == "Novita" and not NOVITA_API_KEY:
    raise ValueError("NOVITA_API_KEY not found in environment variables")
elif args.provider == "Together" and not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")
elif args.provider == "Fireworks" and not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in environment variables")
