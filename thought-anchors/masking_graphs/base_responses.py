#!/usr/bin/env python3
"""
Base response generation for MMLU questions using Fireworks API with logprobs.

This module processes MMLU questions asynchronously, extracting logprobs for
answer choices (A, B, C, D) to compute likelihood of each option.
"""

import asyncio
import json
import math
import os
import time
import csv
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

from resample.provider_config import ProviderConfig
from resample.fireworks_logprobs import gen_with_config
from resample.rollouts import Rollouts, FwResponse
from load_mmlu import load_all_mmlu, create_mmlu_prompt
from copy import deepcopy


def extract_answer_logprobs(response: FwResponse) -> Dict[str, Any]:
    """
    Extract logprobs for answer tokens (A, B, C, D) from a response.

    Improved version that looks for answer tokens in the top alternatives
    even when the model doesn't directly generate them.

    Args:
        response: FwResponse object containing logprobs data

    Returns:
        Dictionary with:
        - generated_answer: The actual answer generated (A/B/C/D)
        - answer_logprob: Logprob of the generated answer
        - all_answer_probs: Dict mapping A/B/C/D to their probabilities
        - raw_logprobs: Dict mapping A/B/C/D to their raw logprobs
        - no_answer_generated: True if model didn't generate a letter
    """
    result = {
        "generated_answer": None,
        "answer_logprob": None,
        "all_answer_probs": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "raw_logprobs": {
            "A": -float("inf"),
            "B": -float("inf"),
            "C": -float("inf"),
            "D": -float("inf"),
        },
        "no_answer_generated": False,
        "error": None,
    }

    if not response.logprobs:
        result["error"] = "No logprobs in response"
        return result

    # Get the generated text - with new prompt format, answer is at the end after "Answer:"
    generated_text = response.text.strip()

    # Look for answer at the end of the text (after "Final answer:" or "Answer:")
    if "Final Answer:" in generated_text:
        # Get text after the last "Final answer:"
        answer_part = generated_text.split("Final Answer:")[-1].strip()
        # Remove markdown bold formatting if present
        answer_part_clean = answer_part.replace("**", "").strip()

        if answer_part_clean and answer_part_clean[0] in "ABCD":
            result["generated_answer"] = answer_part_clean[0]
        elif (
            answer_part_clean
            and len(answer_part_clean) >= 2
            and answer_part_clean[0] == "["
            and answer_part_clean[1] in "ABCD"
        ):
            # Handle bracket format like "[A"
            result["generated_answer"] = answer_part_clean[1]
    elif "Answer:" in generated_text:
        # Fallback for old format
        answer_part = generated_text.split("Answer:")[-1].strip()
        # Remove markdown bold formatting if present
        answer_part_clean = answer_part.replace("**", "").strip()

        if answer_part_clean and answer_part_clean[0] in "ABCD":
            result["generated_answer"] = answer_part_clean[0]
        elif (
            answer_part_clean
            and len(answer_part_clean) >= 2
            and answer_part_clean[0] == "["
            and answer_part_clean[1] in "ABCD"
        ):
            result["generated_answer"] = answer_part_clean[1]
    elif generated_text and generated_text[0] in "ABCD":
        # Fallback: check if text starts with answer (very old format)
        result["generated_answer"] = generated_text[0]
    elif (
        generated_text
        and len(generated_text) >= 2
        and generated_text[0] == "["
        and generated_text[1] in "ABCD"
    ):
        # Handle case where response starts with "[A", "[B", etc.
        result["generated_answer"] = generated_text[1]

    # Find the answer token in the token list
    tokens = response.logprobs.tokens
    token_logprobs = response.logprobs.token_logprobs
    top_logprobs = response.logprobs.top_logprobs
    text_offset = response.logprobs.text_offset
    prompt_length = response.logprobs.prompt_length

    # Look for "Final Answer:" or "Answer:" in tokens to find where the answer should be
    answer_token_idx = None
    for i in range(len(tokens) - 1, -1, -1):  # Search backwards
        # Check for "Final Answer:" pattern
        if (
            i + 2 < len(tokens)
            and tokens[i] == "Final"
            and tokens[i + 1] == " Answer"
            and tokens[i + 2] == ":"
        ):
            # The answer token should be after "Final Answer:"
            answer_token_idx = i + 3  # Skip "Final", " Answer", and ":"
            # Skip spaces, markdown asterisks and brackets to get to the actual letter
            while answer_token_idx < len(tokens):
                token = tokens[answer_token_idx].strip()
                # Skip empty tokens (spaces), brackets, and asterisks
                if token == "" or token in ["[", "*", "**", "***", "****"]:
                    answer_token_idx += 1
                else:
                    break
            break
        # Check for old "Answer:" pattern
        elif i + 1 < len(tokens) and tokens[i] == "Answer" and tokens[i + 1] == ":":
            # The answer token should be after "Answer:"
            answer_token_idx = i + 2  # Skip "Answer" and ":"
            # Skip spaces, markdown asterisks and brackets to get to the actual letter
            while answer_token_idx < len(tokens):
                token = tokens[answer_token_idx].strip()
                # Skip empty tokens (spaces), brackets, and asterisks
                if token == "" or token in ["[", "*", "**", "***", "****"]:
                    answer_token_idx += 1
                else:
                    break
            break

    # Fallback: use prompt_length method if neither pattern found
    if answer_token_idx is None:
        for i, offset in enumerate(text_offset):
            if offset >= prompt_length:
                answer_token_idx = i
                # Skip spaces, markdown asterisks and brackets to get to the actual letter
                while answer_token_idx < len(tokens):
                    token = tokens[answer_token_idx].strip()
                    # Skip empty tokens (spaces), brackets, and asterisks
                    if token == "" or token in ["[", "*", "**", "***", "****"]:
                        answer_token_idx += 1
                    else:
                        break
                break

    if answer_token_idx is None or answer_token_idx >= len(tokens):
        result["error"] = "Could not find answer token index"
        return result

    # Get the answer token and its logprob
    answer_token = tokens[answer_token_idx]
    answer_logprob = token_logprobs[answer_token_idx]

    # Check if we can extract answer probs from the top alternatives
    if answer_token_idx < len(top_logprobs):
        top_probs = top_logprobs[answer_token_idx]

        # Look for A, B, C, D tokens in the top alternatives
        for token, logprob in top_probs.items():
            token_clean = token.strip()
            if token_clean and token_clean[0] in "ABCD":
                letter = token_clean[0]
                result["raw_logprobs"][letter] = logprob
                result["all_answer_probs"][letter] = math.exp(logprob)

        # Also check if the actually generated token is an answer
        answer_token_clean = answer_token.strip()
        if answer_token_clean and answer_token_clean[0] in "ABCD":
            letter = answer_token_clean[0]
            result["answer_logprob"] = answer_logprob
            result["raw_logprobs"][letter] = answer_logprob
            result["all_answer_probs"][letter] = math.exp(answer_logprob)
        else:
            # Model didn't generate a letter directly
            result["no_answer_generated"] = True

    # Normalize probabilities to sum to 1 (for the tokens we found)
    total_prob = sum(result["all_answer_probs"].values())
    if total_prob > 0:
        for letter in "ABCD":
            result["all_answer_probs"][letter] /= total_prob

    return result


async def process_single_question(
    question_data: Dict[str, Any], config: ProviderConfig, semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Process a single MMLU question and extract answer logprobs.

    Args:
        question_data: MMLU question dictionary
        config: Provider configuration
        semaphore: Semaphore for rate limiting

    Returns:
        Dictionary with question, response, and extracted logprobs
    """
    async with semaphore:
        # Create prompt
        prompt = create_mmlu_prompt(question_data, config.model)
        # print(config)
        # quit()

        # try:
        # Generate response with logprobs
        rollout = await gen_with_config(
            prompt=prompt, num_responses=1, config=config, return_dataclass=True
        )

        if not rollout.responses:
            return {
                "question_data": question_data,
                "error": "No responses generated",
                "prompt": prompt,
            }

        response = rollout.responses[0]

        # Extract answer logprobs
        logprobs_data = extract_answer_logprobs(response)
        # print(f"{logprobs_data=}")
        if logprobs_data["generated_answer"] is None:
            config_copy = deepcopy(config)
            config_copy.max_tokens = 10
            rollout = await gen_with_config(
                prompt=prompt, num_responses=1, config=config_copy, return_dataclass=True
            )
            logprobs_data = extract_answer_logprobs(rollout.responses[0])

        # Compile result
        result = {
            "question_data": question_data,
            "prompt": prompt,
            "generated_text": response.text,
            "correct_answer": question_data["answer_letter"],
            "generated_answer": logprobs_data["generated_answer"],
            "is_correct": logprobs_data["generated_answer"] == question_data["answer_letter"],
            "answer_probs": logprobs_data["all_answer_probs"],
            "raw_logprobs": logprobs_data["raw_logprobs"],
            "no_answer_generated": logprobs_data.get("no_answer_generated", False),
            "logprob_error": logprobs_data.get("error"),
            "usage": (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
        }

        return result

        # except Exception as e:
        #     return {"question_data": question_data, "error": str(e), "prompt": prompt}


async def process_mmlu_batch(
    questions: List[Dict[str, Any]],
    model: str = "qwen3-30b-a3b",
    max_concurrent: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 1,
    verbose: bool = True,
    req_exist: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process a batch of MMLU questions asynchronously.

    Args:
        questions: List of MMLU question dictionaries
        model: Model to use (default: qwen3-30b-a3b)
        max_concurrent: Maximum concurrent API requests
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Max tokens to generate (10 to handle bracket format)
        verbose: Whether to show progress

    Returns:
        List of results with answers and logprobs
    """
    # Configure the model
    config = ProviderConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        logprobs=5,  # Get top 5 alternatives
        echo=True,  # Need this to identify where prompt ends
        verbose=False,  # Individual request verbosity
        req_exist=req_exist,
        max_retries=6,
    )

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [process_single_question(q, config, semaphore) for q in questions]

    # Process with progress bar if verbose
    if verbose:
        results = []
        for task in async_tqdm.as_completed(tasks, desc="Processing MMLU questions"):
            result = await task
            results.append(result)

    else:
        results = await asyncio.gather(*tasks)

    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the results from MMLU processing.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with statistics and analysis
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    errors = sum(1 for r in results if "error" in r)

    # Analyze probability distributions
    correct_answer_probs = []
    generated_answer_probs = []

    for r in results:
        if "answer_probs" in r and r["answer_probs"]:
            correct_letter = r["question_data"]["answer_letter"]
            if correct_letter in r["answer_probs"]:
                correct_answer_probs.append(r["answer_probs"][correct_letter])

            generated_letter = r.get("generated_answer")
            if generated_letter and generated_letter in r["answer_probs"]:
                generated_answer_probs.append(r["answer_probs"][generated_letter])

    analysis = {
        "total_questions": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "errors": errors,
        "error_rate": errors / total if total > 0 else 0,
        "avg_correct_answer_prob": np.mean(correct_answer_probs) if correct_answer_probs else 0,
        "avg_generated_answer_prob": (
            np.mean(generated_answer_probs) if generated_answer_probs else 0
        ),
        "median_correct_answer_prob": (
            np.median(correct_answer_probs) if correct_answer_probs else 0
        ),
    }

    # Breakdown by subject if available
    subject_stats = {}
    for r in results:
        subject = r["question_data"].get("subject", "unknown")
        if subject not in subject_stats:
            subject_stats[subject] = {"total": 0, "correct": 0}
        subject_stats[subject]["total"] += 1
        if r.get("is_correct", False):
            subject_stats[subject]["correct"] += 1

    for subject, stats in subject_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    analysis["by_subject"] = subject_stats

    return analysis


async def generate_all_mmlu(
    model: str = "qwen3-30b-a3b",
    max_questions: int = None,
    batch_size: int = 100,
    max_concurrent: int = 10,
    output_dir: str = "results/mmlu_results",
    temperature: float = 0.0,
    req_exist: bool = False,
):
    """
    Generate responses for all MMLU questions.

    Args:
        model: Model to use
        max_questions: Limit number of questions (None for all)
        batch_size: Process questions in batches of this size
        max_concurrent: Max concurrent API requests per batch
        output_dir: Directory to save results
        temperature: Sampling temperature
    """
    # Load all MMLU questions
    print("Loading MMLU data...")
    all_questions = load_all_mmlu()
    # print(all_questions[0])
    # quit()

    if max_questions:
        all_questions = all_questions[:max_questions]

    print(f"Processing {len(all_questions)} questions in batches of {batch_size}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    # Process in batches
    for i in range(0, len(all_questions), batch_size):
        batch = all_questions[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_questions) + batch_size - 1) // batch_size

        print(f"\n=== Batch {batch_num}/{total_batches} ===")

        batch_results = await process_mmlu_batch(
            batch,
            model=model,
            max_concurrent=max_concurrent,
            temperature=temperature,
            verbose=True,
            req_exist=req_exist,
        )

        all_results.extend(batch_results)

    # Final analysis
    analysis = analyze_results(all_results)

    # Save complete results
    final_file = Path(output_dir) / "all_results.json"
    with open(final_file, "w") as f:
        json.dump(
            {
                "model": model,
                "total_questions": len(all_results),
                "analysis": analysis,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n=== Final Results ===")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print(f"Results saved to {final_file}")

    return all_results, analysis


def display_detailed_statistics(results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    """
    Display comprehensive statistics about MMLU processing results.

    Args:
        results: List of result dictionaries from generate_all_mmlu
        analysis: Analysis dictionary from analyze_results
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "MMLU PROCESSING STATISTICS")
    print("=" * 80)

    # Overall Performance
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Questions:     {analysis['total_questions']:,}")
    print(f"Correct Answers:     {analysis['correct']:,}")
    print(f"Accuracy:            {analysis['accuracy']:.2%}")
    print(f"Errors:              {analysis['errors']:,} ({analysis['error_rate']:.2%})")

    # Probability Analysis
    print("\n📈 PROBABILITY ANALYSIS")
    print("-" * 40)
    print(f"Avg Prob (Correct):  {analysis['avg_correct_answer_prob']:.4f}")
    print(f"Avg Prob (Generated):{analysis['avg_generated_answer_prob']:.4f}")
    print(f"Median Prob (Correct): {analysis['median_correct_answer_prob']:.4f}")

    # Answer Distribution
    print("\n🎯 ANSWER DISTRIBUTION")
    print("-" * 40)
    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "None": 0}
    for r in results:
        ans = r.get("generated_answer")
        if ans in answer_counts:
            answer_counts[ans] += 1
        else:
            answer_counts["None"] += 1

    for letter in ["A", "B", "C", "D", "None"]:
        count = answer_counts[letter]
        pct = (count / len(results) * 100) if results else 0
        bar = "█" * int(pct / 2)
        print(f"  {letter:4s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Subject Performance (top 10 by number of questions)
    print("\n📚 SUBJECT PERFORMANCE")  # (Top 10 by Question Count)")
    print("-" * 40)
    print(f"{'Subject':<40} {'Questions':>10} {'Correct':>10} {'Accuracy':>10}")
    print("-" * 70)

    # Sort subjects by number of questions
    sorted_subjects = sorted(
        analysis["by_subject"].items(), key=lambda x: x[1]["total"], reverse=True
    )

    for subject, stats in sorted_subjects:
        subject_display = subject[:37] + "..." if len(subject) > 40 else subject
        print(
            f"{subject_display:<40} {stats['total']:>10} {stats['correct']:>10} {stats['accuracy']:>9.1%}"
        )

    # Performance by Confidence Bins
    print("\n🎰 PERFORMANCE BY CONFIDENCE")
    print("-" * 40)

    # Create confidence bins
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    bin_stats = {bin_range: {"total": 0, "correct": 0} for bin_range in bins}

    for r in results:
        if "answer_probs" in r and r.get("generated_answer"):
            prob = r["answer_probs"].get(r["generated_answer"], 0)
            for bin_range in bins:
                if bin_range[0] <= prob < bin_range[1]:
                    bin_stats[bin_range]["total"] += 1
                    if r.get("is_correct", False):
                        bin_stats[bin_range]["correct"] += 1
                    break

    print(f"{'Confidence Range':<20} {'Questions':>10} {'Correct':>10} {'Accuracy':>10}")
    for bin_range, stats in bin_stats.items():
        range_str = f"{bin_range[0]:.0%}-{bin_range[1]:.0%}"
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{range_str:<20} {stats['total']:>10} {stats['correct']:>10} {acc:>9.1%}")

    # Processing Issues
    print("\n⚠️  PROCESSING ISSUES")
    print("-" * 40)

    no_answer_count = sum(1 for r in results if r.get("no_answer_generated", False))
    logprob_error_count = sum(1 for r in results if r.get("logprob_error"))

    print(f"No Answer Generated: {no_answer_count:,} ({no_answer_count/len(results)*100:.1f}%)")
    print(
        f"Logprob Errors:      {logprob_error_count:,} ({logprob_error_count/len(results)*100:.1f}%)"
    )
    print(f"API Errors:          {analysis['errors']:,} ({analysis['error_rate']:.1%})")

    # Token Usage Statistics
    print("\n💰 TOKEN USAGE")
    print("-" * 40)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for r in results:
        if r.get("usage"):
            total_prompt_tokens += r["usage"]["prompt_tokens"]
            total_completion_tokens += r["usage"]["completion_tokens"]
            total_tokens += r["usage"]["total_tokens"]

    avg_prompt = total_prompt_tokens / len(results) if results else 0
    avg_completion = total_completion_tokens / len(results) if results else 0
    avg_total = total_tokens / len(results) if results else 0

    print(f"Total Tokens Used:   {total_tokens:,}")
    print(f"  - Prompt Tokens:   {total_prompt_tokens:,}")
    print(f"  - Completion:      {total_completion_tokens:,}")
    print(f"Average per Question:")
    print(f"  - Prompt:          {avg_prompt:.1f}")
    print(f"  - Completion:      {avg_completion:.1f}")
    print(f"  - Total:           {avg_total:.1f}")

    print("\n" + "=" * 80)


def export_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Export MMLU results to CSV for easy filtering and analysis.

    Args:
        results: List of result dictionaries from generate_all_mmlu
        output_path: Path where to save the CSV file
    """
    csv_path = Path(output_path)

    # Define CSV headers
    headers = [
        "hash_key",
        "global_index",
        "subject",
        "split",
        "question",
        "choices",
        "correct_answer",
        "generated_answer",
        "is_correct",
        "prob_A",
        "prob_B",
        "prob_C",
        "prob_D",
        "prob_correct",
        "prob_generated",
        "no_answer_generated",
        "error",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for r in results:
            q_data = r.get("question_data", {})
            answer_probs = r.get("answer_probs", {})
            usage = r.get("usage", {})

            row = {
                "hash_key": q_data.get("hash_key", ""),
                "global_index": q_data.get("global_index", ""),
                "subject": q_data.get("subject", ""),
                "split": q_data.get("split", ""),
                "question": q_data.get("question", ""),
                "choices": "|".join(q_data.get("choices", [])),  # Join choices with pipe
                "correct_answer": q_data.get("answer_letter", ""),
                "generated_answer": r.get("generated_answer", ""),
                "is_correct": r.get("is_correct", False),
                "prob_A": answer_probs.get("A", 0.0),
                "prob_B": answer_probs.get("B", 0.0),
                "prob_C": answer_probs.get("C", 0.0),
                "prob_D": answer_probs.get("D", 0.0),
                "prob_correct": answer_probs.get(q_data.get("answer_letter", ""), 0.0),
                "prob_generated": answer_probs.get(r.get("generated_answer", ""), 0.0),
                "no_answer_generated": r.get("no_answer_generated", False),
                "error": r.get("error", ""),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            writer.writerow(row)

    print(f"CSV results exported to {csv_path}")


def load_incorrect_questions_from_csv(csv_path: str) -> List[str]:
    """
    Load hash keys of incorrect questions from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of hash keys for questions that were answered incorrectly
    """
    incorrect_hashes = []

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["is_correct"].lower() == "false":
                incorrect_hashes.append(row["hash_key"])

    return incorrect_hashes


def load_json_result_analysis(
    json_path: str = "results/mmlu_results/all_results.json",
) -> List[Dict[str, Any]]:
    """
    Load JSON file into a list of dictionaries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    analysis = data["analysis"]
    return results, analysis


if __name__ == "__main__":
    # Configuration constants (uppercase for global constants)
    MAX_QUESTIONS = 16000  # Set to None to process all questions
    MAX_CONCURRENT = 250
    BATCH_SIZE = 6000
    OUTPUT_DIR = "results/mmlu_results"
    MODEL = "qwen3-30b-a3b"
    # MODEL = "Qwen/Qwen2.5-14B"
    # MODEL = "llama-v3p3-70b-instruct"  # fireworks
    # MODEL = r"meta-llama/Llama-3.3-70B-Instruct-Turbo"  # together
    MODEL_STR = MODEL.split("/")[-1]
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_STR)
    # MAX_QUESTIONS = 1000
    TEMPERATURE = 0.0

    # # Run the main generation function
    results, analysis = asyncio.run(
        generate_all_mmlu(
            model=MODEL,
            max_questions=MAX_QUESTIONS,
            batch_size=BATCH_SIZE,
            max_concurrent=MAX_CONCURRENT,
            output_dir=OUTPUT_DIR,
            temperature=TEMPERATURE,
            req_exist=False,
        )
    )

    # results, analysis = load_json_result_analysis()

    # Display detailed statistics
    display_detailed_statistics(results, analysis)

    # Export results to CSV
    csv_file = Path(OUTPUT_DIR) / f"results_{MAX_QUESTIONS}.csv"
    print(f"Exporting results to: {csv_file}")
    export_results_to_csv(results, str(csv_file))
