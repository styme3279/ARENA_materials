#!/usr/bin/env python3
"""
Base response generation for MMLU questions using thinking mode.

This module processes MMLU questions using thinking mode with multiple runs
per question since thinking can have variance in the reasoning process.
"""

import asyncio
import json
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm
from collections import Counter

from resample.openrouter_clean import generate_responses_openrouter
from load_mmlu import (
    load_all_mmlu,
    create_mmlu_prompt,
    load_mmlu_below_accuracy_threshold,
)
from pkld import pkld


def get_parameter_specific_output_dir(
    max_questions: int,
    model: str,
    max_nonthinking_accuracy: Optional[float] = None,
    base_dir: str = "results/mmlu_thinking_results",
) -> str:
    """
    Generate a parameter-specific output directory name.

    Args:
        max_questions: Maximum number of questions to process
        max_nonthinking_accuracy: Maximum accuracy threshold for non-thinking mode
        base_dir: Base directory for results

    Returns:
        Parameter-specific output directory path
    """
    acc_str = (
        f"acc{max_nonthinking_accuracy:.1f}"
        if max_nonthinking_accuracy is not None
        else "accNone"
    )
    model_latter = model.split("/")[-1].split("-")[0]
    return f"{base_dir}/q{max_questions}_{acc_str}/{model_latter}"


def extract_answer_from_thinking(response_text: str) -> Optional[str]:
    """
    Extract answer from thinking mode response.

    Looking for pattern: "Final Answer: [LETTER]"

    Args:
        response_text: The generated text from the model

    Returns:
        The extracted answer letter (A/B/C/D) or None if not found
    """
    # Remove the </think> tag if present
    text = response_text.replace("</think>", "").strip()

    # Look for "Final Answer: X" pattern (case insensitive)
    # Try multiple patterns to be robust
    patterns = [
        # Primary "Final Answer" patterns
        r"Final Answer:\s*([A-D])",  # "Final Answer: A"
        r"Final answer:\s*([A-D])",  # "Final answer: A"
        r"FINAL ANSWER:\s*([A-D])",  # "FINAL ANSWER: A"
        r"Final Answer:\s*\[([A-D])\]",  # "Final Answer: [A]"
        r"Final answer:\s*\[([A-D])\]",  # "Final answer: [A]"
        r"Final Answer:\s*\[([A-D])",  # "Final Answer: [A" (bracket not closed)
        r"Final answer:\s*\[([A-D])",  # "Final answer: [A" (bracket not closed)
        # Common answer patterns (for truncated traces)
        r"answer is ([A-D])",  # "answer is B"
        r"answer:\s*([A-D])",  # "answer: C"
        r"answer:\s*\[([A-D])",  # "answer: [C" (bracket not closed)
        r"answer is\s*\[([A-D])",  # "answer is [B" (bracket not closed)
        r"Therefore,?\s+answer\s+([A-D])",  # "Therefore, answer D"
        r"So,?\s+answer\s+([A-D])",  # "So, answer A"
        r"correct answer is ([A-D])",  # "correct answer is B"
        r"The answer is ([A-D])",  # "The answer is C"
        r"The answer is\s*\[([A-D])",  # "The answer is [C" (bracket not closed)
        # LaTeX boxed notation (for mathematical formatting)
        r"\\boxed\{([A-D])\}",  # LaTeX boxed notation
        r"\$\$\\boxed\{([A-D])\}\$\$",  # LaTeX with dollar signs
        # Period-terminated patterns
        r"answer[:\s]+([A-D])\.",  # "answer B." or "answer: B."
        r"Therefore,?\s+answer\s+([A-D])\.",  # "Therefore, answer B."
        # More flexible patterns
        r"the\s+correct\s+answer\s+is[:\s]*([A-D])",  # "the correct answer is B"
        r"Thus,?\s+the\s+(?:correct\s+)?answer\s+is[:\s]*([A-D])",  # "Thus, the answer is: B"
        r"option\s+([A-D])\s+is\s+(?:the\s+)?correct",  # "option B is correct"
        r"the\s+correct\s+choice\s+is\s+([A-D])",  # "the correct choice is D"
        r"choose\s+([A-D])",  # "choose B"
        r"select\s+([A-D])",  # "select B"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Enhanced fallback: Look for answer in the last 200 characters
    # This helps with truncated traces
    text_end = text[-200:] if len(text) > 200 else text

    # Check for answer letter after conclusive words
    fallback_patterns = [
        r"(?:therefore|thus|so|hence|consequently|finally)[^A-D]*([A-D])(?:\.|,|\s|$|\))",
        r"([A-D])\s+is\s+(?:the\s+)?(?:correct|right|best)",
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, text_end, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Final fallback: Only if text ends with a letter preceded by answer-related word
    if text:
        text_end = text[-50:] if len(text) > 50 else text
        match = re.search(
            r"(?:answer|correct|choose|select)[^A-D]*([A-D])(?:\s|$|\.|,)",
            text_end,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

    return None


async def process_single_question_once(
    question_data: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    DEPRECATED: This function is no longer used. Use process_single_question_multiple instead.

    Process a single MMLU question once with thinking mode.

    NOTE: This function always uses seed=0, which causes duplicate responses when called
    multiple times. process_single_question_multiple now handles multiple responses
    correctly by using num_responses parameter with proper seed variation.

    Args:
        question_data: MMLU question dictionary
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        semaphore: Semaphore for rate limiting

    Returns:
        Dictionary with question, response, and extracted answer
    """
    async with semaphore:
        # Create prompt with thinking mode enabled and just_prompt=True
        prompt = create_mmlu_prompt(
            question_data, model, enable_thinking=True, just_prompt=True
        )

        try:
            # Generate response using clean openrouter (returns Rollouts dataclass)
            rollouts = await generate_responses_openrouter(
                prompt=prompt,
                num_responses=1,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                verbose=False,
            )

            if not rollouts.responses or len(rollouts.responses) == 0:
                return {
                    "question_data": question_data,
                    "error": "No responses generated",
                    "prompt": prompt,
                }

            response = rollouts.get_first_response()

            # Check for error in response
            if response.finish_reason == "error":
                return {
                    "question_data": question_data,
                    "error": response.text,
                    "prompt": prompt,
                }

            # Extract answer from thinking response
            # Use the full text which combines reasoning and content
            response_text = response.text
            generated_answer = extract_answer_from_thinking(response_text)

            # Compile result
            result = {
                "question_data": question_data,
                "prompt": prompt,
                "generated_text": response_text,
                "reasoning": response.reasoning,  # Store reasoning separately
                "post_content": response.post,  # Store final answer separately
                "correct_answer": question_data["answer_letter"],
                "generated_answer": generated_answer,
                "is_correct": generated_answer
                == question_data["answer_letter"],
                "extraction_failed": generated_answer is None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

            return result

        except Exception as e:
            return {
                "question_data": question_data,
                "error": str(e),
                "prompt": prompt,
            }


async def process_single_question_multiple(
    question_data: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """
    Process a single MMLU question multiple times and aggregate results.

    Args:
        question_data: MMLU question dictionary
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        semaphore: Semaphore for rate limiting
        num_runs: Number of times to run each question

    Returns:
        Dictionary with aggregated results from multiple runs
    """
    async with semaphore:
        # Create prompt with thinking mode enabled and just_prompt=True
        prompt = create_mmlu_prompt(
            question_data, model, enable_thinking=True, just_prompt=True
        )

        try:
            # Generate multiple responses in a single API call with different seeds
            rollouts = await generate_responses_openrouter(
                prompt=prompt,
                num_responses=num_runs,  # Use num_runs for proper seed variation
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                verbose=False,
            )

            if not rollouts.responses or len(rollouts.responses) == 0:
                return {
                    "question_data": question_data,
                    "error": "No responses generated",
                    "prompt": prompt,
                }

        except Exception as e:
            return {
                "question_data": question_data,
                "error": str(e),
                "prompt": prompt,
            }

        # Process all responses from the rollouts
        generated_answers = []
        thinking_traces = []
        all_is_correct = []  # Track whether each response is correct
        errors = []
        extraction_failures = 0
        reasoning_traces = []
        post_contents = []
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        correct_answer = question_data["answer_letter"]

        for response in rollouts.responses:
            # Check for error in response
            if response.finish_reason == "error":
                errors.append(response.text)
                all_is_correct.append(False)  # Error responses are incorrect
                continue

            # Extract answer from thinking response
            response_text = response.text
            generated_answer = extract_answer_from_thinking(response_text)

            if generated_answer:
                generated_answers.append(generated_answer)
                # Check if this individual response is correct
                all_is_correct.append(generated_answer == correct_answer)
            else:
                extraction_failures += 1
                all_is_correct.append(False)  # Failed extraction is incorrect

            thinking_traces.append(response_text)

            # Collect reasoning and post content if available
            if response.reasoning:
                reasoning_traces.append(response.reasoning)
            if response.post:
                post_contents.append(response.post)

        # Aggregate usage from all responses
        for response in rollouts.responses:
            if response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage[
                    "completion_tokens"
                ] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens

    # Determine the majority answer
    if generated_answers:
        answer_counts = Counter(generated_answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        majority_count = answer_counts[majority_answer]
        confidence = majority_count / len(generated_answers)
    else:
        majority_answer = None
        confidence = 0.0
        answer_counts = {}

    correct_answer = question_data["answer_letter"]

    # Calculate thinking trace statistics
    thinking_lengths = [len(trace) for trace in thinking_traces if trace]
    avg_thinking_length = np.mean(thinking_lengths) if thinking_lengths else 0

    # Compile aggregated result
    aggregated_result = {
        "question_data": question_data,
        "num_runs": num_runs,
        "successful_runs": len(generated_answers),
        "errors": errors,
        "extraction_failures": extraction_failures,
        "correct_answer": correct_answer,
        "majority_answer": majority_answer,
        "is_correct": majority_answer == correct_answer,
        "confidence": confidence,
        "answer_distribution": dict(answer_counts),
        "all_generated_answers": generated_answers,
        "all_is_correct": all_is_correct,  # List of whether each response was correct
        "sample_thinking_trace": (
            thinking_traces[0] if thinking_traces else None
        ),
        "all_thinking_traces": thinking_traces,  # Store all traces for analysis
        "avg_thinking_length": avg_thinking_length,
        "reasoning": (
            reasoning_traces[0] if reasoning_traces else None
        ),  # Sample reasoning
        "post_content": (
            post_contents[0] if post_contents else None
        ),  # Sample post content
        "total_usage": total_usage,
    }

    return aggregated_result


async def process_mmlu_batch_thinking(
    questions: List[Dict[str, Any]],
    model: str = "qwen/qwen3-30b-a3b",
    max_concurrent: int = 10,
    temperature: float = 0.6,  # Moderate temperature for thinking variance
    max_tokens: int = 8192,  # Increased to prevent truncation of thinking traces
    num_runs_per_question: int = 10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a batch of MMLU questions with thinking mode.

    Args:
        questions: List of MMLU question dictionaries
        model: Model to use (default: qwen/qwen3-30b-a3b)
        max_concurrent: Maximum concurrent API requests
        temperature: Sampling temperature (default: 0.6 for moderate variance)
        max_tokens: Max tokens to generate (default: 4096 for extended thinking)
        num_runs_per_question: Number of times to run each question
        verbose: Whether to show progress

    Returns:
        List of aggregated results with majority voting
    """
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [
        process_single_question_multiple(
            q, model, temperature, max_tokens, semaphore, num_runs_per_question
        )
        for q in questions
    ]

    # Process with progress bar if verbose
    if verbose:
        results = []
        desc = f"Processing MMLU questions ({num_runs_per_question}x each)"
        for task in async_tqdm.as_completed(tasks, desc=desc):
            result = await task
            results.append(result)
    else:
        results = await asyncio.gather(*tasks)

    return results


def analyze_thinking_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the results from MMLU thinking mode processing.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with statistics and analysis
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    errors = sum(1 for r in results if r.get("errors"))

    # Analyze confidence levels
    confidences = [r.get("confidence", 0) for r in results]
    correct_confidences = [
        r.get("confidence", 0) for r in results if r.get("is_correct")
    ]
    incorrect_confidences = [
        r.get("confidence", 0) for r in results if not r.get("is_correct")
    ]

    # Analyze extraction failures
    total_extraction_failures = sum(
        r.get("extraction_failures", 0) for r in results
    )

    analysis = {
        "total_questions": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "errors": errors,
        "error_rate": errors / total if total > 0 else 0,
        "avg_confidence": np.mean(confidences) if confidences else 0,
        "avg_correct_confidence": (
            np.mean(correct_confidences) if correct_confidences else 0
        ),
        "avg_incorrect_confidence": (
            np.mean(incorrect_confidences) if incorrect_confidences else 0
        ),
        "total_extraction_failures": total_extraction_failures,
        "avg_extraction_failures_per_question": (
            total_extraction_failures / total if total > 0 else 0
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
        stats["accuracy"] = (
            stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        )

    analysis["by_subject"] = subject_stats

    return analysis


async def generate_all_mmlu_thinking(
    model: str = "qwen/qwen3-30b-a3b",
    max_questions: int = 1000,
    batch_size: int = 50,
    max_concurrent: int = 5,
    output_dir: str = "results/mmlu_thinking_results",
    temperature: float = 0.6,
    max_tokens: int = 8192,
    num_runs_per_question: int = 10,
    max_nonthinking_accuracy: float = 0.5,
):
    """
    Generate thinking mode responses for all MMLU questions.

    Args:
        model: Model to use
        max_questions: Limit number of questions (None for all)
        batch_size: Process questions in batches of this size
        max_concurrent: Max concurrent API requests
        output_dir: Directory to save results
        temperature: Sampling temperature
        max_tokens: Max tokens for thinking
        num_runs_per_question: Number of runs per question for majority voting
        max_nonthinking_accuracy: If not None, only process questions where the non-thinking
                                  model had prob_correct <= this value (default 0.5).
                                  Set to None to process all questions.
    """
    # Load MMLU questions based on accuracy threshold
    if max_nonthinking_accuracy is not None:
        print(
            f"Loading MMLU questions with prob_correct <= {max_nonthinking_accuracy}..."
        )
        all_questions = load_mmlu_below_accuracy_threshold(
            max_questions=max_questions,
            max_prob_correct=max_nonthinking_accuracy,
        )
    else:
        print("Loading all MMLU questions...")
        all_questions = load_all_mmlu()
        if max_questions:
            all_questions = all_questions[:max_questions]

    # print(f"Processing {len(all_questions)} questions in batches of {batch_size}")
    # print(f"Each question will be run {num_runs_per_question} times with thinking mode")
    # print(f"{max_nonthinking_accuracy=}")
    # quit()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    # Process in batches
    for i in range(0, len(all_questions), batch_size):
        batch = all_questions[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_questions) + batch_size - 1) // batch_size

        print(f"\n=== Batch {batch_num}/{total_batches} ===")

        batch_results = await process_mmlu_batch_thinking(
            batch,
            model=model,
            max_concurrent=max_concurrent,
            temperature=temperature,
            max_tokens=max_tokens,
            num_runs_per_question=num_runs_per_question,
            verbose=True,
        )

        all_results.extend(batch_results)

        # Save intermediate results
        intermediate_file = Path(output_dir) / f"batch_{batch_num:04d}.json"
        with open(intermediate_file, "w") as f:
            json.dump(batch_results, f, indent=2)

        print(f"Saved batch {batch_num} to {intermediate_file}")

    # Final analysis
    analysis = analyze_thinking_results(all_results)

    # Save complete results
    final_file = Path(output_dir) / "all_results.json"
    with open(final_file, "w") as f:
        json.dump(
            {
                "model": model,
                "total_questions": len(all_results),
                "num_runs_per_question": num_runs_per_question,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "analysis": analysis,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n=== Final Results ===")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print(f"Average Confidence: {analysis['avg_confidence']:.2%}")
    print(f"Results saved to {final_file}")

    return all_results, analysis


def display_thinking_statistics(
    results: List[Dict[str, Any]], analysis: Dict[str, Any]
) -> None:
    """
    Display comprehensive statistics about MMLU thinking mode results.

    Args:
        results: List of result dictionaries
        analysis: Analysis dictionary
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MMLU THINKING MODE STATISTICS")
    print("=" * 80)

    # Overall Performance
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Questions:     {analysis['total_questions']:,}")
    print(f"Correct Answers:     {analysis['correct']:,}")
    print(f"Accuracy:            {analysis['accuracy']:.2%}")
    print(
        f"Errors:              {analysis['errors']:,} ({analysis['error_rate']:.2%})"
    )

    # Confidence Analysis
    print("\n📈 CONFIDENCE ANALYSIS")
    print("-" * 40)
    print(f"Avg Confidence (All):      {analysis['avg_confidence']:.2%}")
    print(
        f"Avg Confidence (Correct):  {analysis['avg_correct_confidence']:.2%}"
    )
    print(
        f"Avg Confidence (Wrong):    {analysis['avg_incorrect_confidence']:.2%}"
    )

    # Run Statistics
    print("\n🔄 RUN STATISTICS")
    print("-" * 40)
    total_runs = sum(r.get("successful_runs", 0) for r in results)
    avg_runs = total_runs / len(results) if results else 0
    print(f"Total Runs:          {total_runs:,}")
    print(f"Avg Runs/Question:   {avg_runs:.1f}")
    print(f"Extraction Failures: {analysis['total_extraction_failures']:,}")
    print(
        f"Avg Failures/Q:      {analysis['avg_extraction_failures_per_question']:.2f}"
    )

    # Answer Distribution
    print("\n🎯 ANSWER DISTRIBUTION")
    print("-" * 40)
    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "None": 0}
    for r in results:
        ans = r.get("majority_answer")
        if ans in answer_counts:
            answer_counts[ans] += 1
        else:
            answer_counts["None"] += 1

    for letter in ["A", "B", "C", "D", "None"]:
        count = answer_counts[letter]
        pct = (count / len(results) * 100) if results else 0
        bar = "█" * int(pct / 2)
        print(f"  {letter:4s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Subject Performance (top 10)
    if "by_subject" in analysis:
        print("\n📚 SUBJECT PERFORMANCE (Top 10)")
        print("-" * 40)
        print(
            f"{'Subject':<40} {'Questions':>10} {'Correct':>10} {'Accuracy':>10}"
        )
        print("-" * 70)

        sorted_subjects = sorted(
            analysis["by_subject"].items(),
            key=lambda x: x[1]["total"],
            reverse=True,
        )[:10]

        for subject, stats in sorted_subjects:
            subject_display = (
                subject[:37] + "..." if len(subject) > 40 else subject
            )
            print(
                f"{subject_display:<40} {stats['total']:>10} "
                f"{stats['correct']:>10} {stats['accuracy']:>9.1%}"
            )

    # Thinking Trace Analysis
    print("\n🧠 THINKING TRACE ANALYSIS")
    print("-" * 40)

    # Collect all thinking traces
    all_traces = []
    trace_lengths = []
    questions_with_traces = 0

    for r in results:
        if "all_thinking_traces" in r and r["all_thinking_traces"]:
            questions_with_traces += 1
            for trace in r["all_thinking_traces"]:
                if trace:
                    all_traces.append(trace)
                    trace_lengths.append(len(trace))
        elif "sample_thinking_trace" in r and r["sample_thinking_trace"]:
            # Fallback for older format
            questions_with_traces += 1
            trace = r["sample_thinking_trace"]
            all_traces.append(trace)
            trace_lengths.append(len(trace))

    if trace_lengths:
        avg_length = np.mean(trace_lengths)
        median_length = np.median(trace_lengths)
        min_length = np.min(trace_lengths)
        max_length = np.max(trace_lengths)
        std_length = np.std(trace_lengths)

        print(f"Questions with traces: {questions_with_traces}/{len(results)}")
        print(f"Total traces collected: {len(all_traces)}")
        print(f"\nTrace Length Statistics (characters):")
        print(f"  Average:    {avg_length:,.0f}")
        print(f"  Median:     {median_length:,.0f}")
        print(f"  Min:        {min_length:,}")
        print(f"  Max:        {max_length:,}")
        print(f"  Std Dev:    {std_length:,.0f}")

        # Check for thinking tags
        traces_with_think_tag = sum(
            1 for t in all_traces if "<think>" in t.lower()
        )
        traces_with_end_think = sum(
            1 for t in all_traces if "</think>" in t.lower()
        )

        print(f"\nThinking Tag Analysis:")
        print(
            f"  With <think>:   {traces_with_think_tag}/{len(all_traces)} ({traces_with_think_tag/len(all_traces)*100:.1f}%)"
        )
        print(
            f"  With </think>:  {traces_with_end_think}/{len(all_traces)} ({traces_with_end_think/len(all_traces)*100:.1f}%)"
        )

        # Check for Final Answer patterns and successful extraction
        traces_with_final = sum(
            1 for t in all_traces if "final answer" in t.lower()
        )
        print(
            f"  With 'Final Answer': {traces_with_final}/{len(all_traces)} ({traces_with_final/len(all_traces)*100:.1f}%)"
        )

        # Check successful extraction rate
        successful_extractions = sum(
            1 for t in all_traces if extract_answer_from_thinking(t) is not None
        )
        print(
            f"  Successfully extracted: {successful_extractions}/{len(all_traces)} ({successful_extractions/len(all_traces)*100:.1f}%)"
        )

        # Sample some trace content for verification
        if all_traces:
            sample_trace = all_traces[0]
            print(f"\nSample trace preview (first 300 chars):")
            print(f"  {sample_trace[:300]}...")

            # Check if traces actually contain reasoning
            avg_words = np.mean(
                [len(t.split()) for t in all_traces[:10]]
            )  # Sample first 10
            print(f"\nAverage word count (sample): {avg_words:.0f} words")

        # Analyze reasoning vs post content split (if available)
        reasoning_lengths = []
        post_lengths = []
        for r in results[:10]:  # Sample first 10
            if "reasoning" in r and r["reasoning"]:
                reasoning_lengths.append(len(r["reasoning"]))
            if "post_content" in r and r["post_content"]:
                post_lengths.append(len(r["post_content"]))

        if reasoning_lengths or post_lengths:
            print(f"\nReasoning vs Post Content Split:")
            if reasoning_lengths:
                print(
                    f"  Avg reasoning length: {np.mean(reasoning_lengths):,.0f} chars"
                )
            if post_lengths:
                print(
                    f"  Avg post content length: {np.mean(post_lengths):,.0f} chars"
                )
    else:
        print("WARNING: No thinking traces found in results!")
        print("This may indicate an issue with trace extraction or generation.")

    # Correlation between trace length and accuracy
    if len(results) > 0:
        print("\n📊 TRACE LENGTH VS ACCURACY")
        print("-" * 40)

        correct_trace_lengths = []
        incorrect_trace_lengths = []

        for r in results:
            if "avg_thinking_length" in r and r["avg_thinking_length"] > 0:
                if r.get("is_correct", False):
                    correct_trace_lengths.append(r["avg_thinking_length"])
                else:
                    incorrect_trace_lengths.append(r["avg_thinking_length"])

        if correct_trace_lengths:
            print(
                f"Avg length (correct):   {np.mean(correct_trace_lengths):,.0f} chars"
            )
        if incorrect_trace_lengths:
            print(
                f"Avg length (incorrect): {np.mean(incorrect_trace_lengths):,.0f} chars"
            )

        if correct_trace_lengths and incorrect_trace_lengths:
            diff = np.mean(correct_trace_lengths) - np.mean(
                incorrect_trace_lengths
            )
            print(
                f"Difference: {diff:+,.0f} chars ({diff/np.mean(incorrect_trace_lengths)*100:+.1f}%)"
            )

    # Token Usage
    print("\n💰 TOKEN USAGE")
    print("-" * 40)

    total_prompt = sum(
        r.get("total_usage", {}).get("prompt_tokens", 0) for r in results
    )
    total_completion = sum(
        r.get("total_usage", {}).get("completion_tokens", 0) for r in results
    )
    total_tokens = sum(
        r.get("total_usage", {}).get("total_tokens", 0) for r in results
    )

    avg_prompt = total_prompt / len(results) if results else 0
    avg_completion = total_completion / len(results) if results else 0
    avg_total = total_tokens / len(results) if results else 0

    print(f"Total Tokens Used:   {total_tokens:,}")
    print(f"  - Prompt:          {total_prompt:,}")
    print(f"  - Completion:      {total_completion:,}")
    print(f"Average per Question:")
    print(f"  - Prompt:          {avg_prompt:.1f}")
    print(f"  - Completion:      {avg_completion:.1f}")
    print(f"  - Total:           {avg_total:.1f}")

    print("\n" + "=" * 80)


@pkld(overwrite=True)
def get_thinking_prompts_async(
    model="qwen/qwen3-30b-a3b",
    temperature=0.6,
    max_tokens=4096,
    num_runs_per_question=10,
    batch_size=1000,
    max_nonthinking_accuracy=0.5,
    max_questions=1000,
    max_concurrent=200,
    get_shortest=False,
):
    """Async version of get_thinking_prompts for use within async contexts."""
    t_start = time.time()
    output_dir = "results/mmlu_thinking_results"
    results, _ = asyncio.run(
        generate_all_mmlu_thinking(
            model=model,
            max_questions=max_questions,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            num_runs_per_question=num_runs_per_question,
            max_nonthinking_accuracy=max_nonthinking_accuracy,
        )
    )

    prompts = []
    for r in results:
        q_formatted = create_mmlu_prompt(
            r["question_data"], model, enable_thinking=True
        )
        num_traces = len(r["all_thinking_traces"])
        n_char_shortest = 100_000
        shortest_trace = None
        for i in range(num_traces):
            thinking_trace = r["all_thinking_traces"][i]
            assert (
                "</think>" in thinking_trace
            ), f"No </think> in {thinking_trace} (No </think> found)"
            post_think_text = thinking_trace.split("</think>")[1]
            num_chars = len(post_think_text)
            if num_chars < n_char_shortest:
                shortest_trace = thinking_trace
                n_char_shortest = num_chars

        if get_shortest:
            prompts.append(q_formatted + shortest_trace)
        else:
            prompts.append(q_formatted + r["all_thinking_traces"][0])

    print(f"get_thinking_prompts took {time.time() - t_start:.1f}s")
    return prompts



@pkld(overwrite=False)
def get_thinking_prompts(
    model="qwen/qwen3-30b-a3b",
    temperature=0.6,
    max_tokens=4096,
    num_runs_per_question=10,
    batch_size=1000,
    max_nonthinking_accuracy=0.5,
    max_questions=1000,
    max_concurrent=200,
    get_shortest=False,
    require_correct=True,
    sanity=0,
):
    """Sync wrapper around get_thinking_prompts_async."""
    t_start = time.time()
    output_dir = "results/mmlu_thinking_results"

    results, _ = asyncio.run(
        generate_all_mmlu_thinking(
            model=model,
            max_questions=max_questions,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            num_runs_per_question=num_runs_per_question,
            max_nonthinking_accuracy=max_nonthinking_accuracy,
        )
    )
    # print("test")
    # quit()
    prompts = []
    q_data_list = []

    for idx_r, r in enumerate(results):
        q_formatted = create_mmlu_prompt(
            r["question_data"],
            model,
            enable_thinking=True,
        )
        num_traces = len(r["all_thinking_traces"])
        n_char_shortest = 100_000
        shortest_trace = None
        if require_correct == -1:
            if all(r["all_is_correct"]):
                continue
        elif require_correct:
            if not any(r["all_is_correct"]):
                continue
        found_good = False
        for i in range(num_traces):
            thinking_trace = r["all_thinking_traces"][i]
            is_correct = r["all_is_correct"][i]
            if require_correct == -1:
                if is_correct:
                    continue
            elif require_correct:
                if not is_correct:
                    continue

            assert (
                "</think>" in thinking_trace
            ), f"No </think> in {thinking_trace} (No </think> found)"
            n_char = len(thinking_trace)
            found_good = True
            if get_shortest:
                if n_char < n_char_shortest:
                    n_char_shortest = n_char
                    shortest_trace = thinking_trace
                else:
                    continue
            else:
                shortest_trace = thinking_trace
                break
        if not found_good:
            continue

        pre_think_close = shortest_trace.split("</think>")[0]
        full_prompt = q_formatted + pre_think_close + "</think>"

        prompts.append(full_prompt)
        q_data_list.append(r["question_data"])
    t_end = time.time()
    print(
        f"Time taken to get {len(prompts)} prompts: {t_end - t_start:.2f} seconds"
    )
    return prompts, q_data_list


def run_all():
    model = r"qwen/qwen3-30b-a3b"
    # model = r"deepseek/deepseek-r1-distill-llama-8b"
    # model = r"deepseek/deepseek-r1-distill-qwen-14b"
    model = r"deepseek/deepseek-r1-distill-llama-70b"
    # model = r"meta-llama/llama-3.3-70b-instruct"
    print(f"Run all: {model=}")
    max_questions = 16000
    # max_questions = 1000
    # max_questions = 2000
    batch_size = 4000
    max_concurrent = 100
    temperature = 0.6
    max_tokens = 4096
    num_runs_per_question = 2
    max_nonthinking_accuracy = None

    # # Generate parameter-specific output directory
    # output_dir = get_parameter_specific_output_dir(
    #     max_questions=16000, max_nonthinking_accuracy=0.5, model=model
    # )

    # results, analysis = asyncio.run(
    #     generate_all_mmlu_thinking(
    #         model=model,
    #         max_questions=max_questions,
    #         batch_size=batch_size,
    #         max_concurrent=max_concurrent,
    #         output_dir=output_dir,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #         num_runs_per_question=10,
    #         max_nonthinking_accuracy=0.5,
    #     )
    # )

    output_dir = get_parameter_specific_output_dir(
        max_questions=16000, max_nonthinking_accuracy=None, model=model
    )

    results, analysis = asyncio.run(
        generate_all_mmlu_thinking(
            model=model,
            max_questions=max_questions,
            batch_size=batch_size,
            max_concurrent=max_concurrent * 2,
            output_dir=output_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            num_runs_per_question=2,
            max_nonthinking_accuracy=None,
        )
    )


# @pkld
def generate_all_mmlu_thinking_wrap(
    model,
    max_questions,
    batch_size,
    max_concurrent,
    output_dir,
    temperature,
    max_tokens,
    num_runs_per_question,
    max_nonthinking_accuracy,
):
    results, analysis = asyncio.run(
        generate_all_mmlu_thinking(
            model=model,
            max_questions=max_questions,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            num_runs_per_question=num_runs_per_question,
            max_nonthinking_accuracy=max_nonthinking_accuracy,
        )
    )
    return results, analysis


if __name__ == "__main__":
    run_all()
    quit()
    # get_thinking_prompts(get_shortest=False)
    # quit()
    # Configuration constants
    MAX_QUESTIONS = 1600  # Use 1000 to match resul_ts_1000.csv
    MAX_CONCURRENT = 200  # Lower concurrency for thinking mode
    BATCH_SIZE = 10000  # Smaller batches since thinking mode is more intensive
    OUTPUT_DIR = "results/mmlu_thinking_results"
    MODEL = "qwen/qwen3-30b-a3b"
    TEMPERATURE = 0.6
    MAX_TOKENS = 4096
    NUM_RUNS_PER_QUESTION = 10  # Number of runs for majority voting
    MAX_NONTHINKING_ACCURACY = 0.5  # Only process questions with prob_correct <= 0.5, set to None for all

    # Run the main generation function
    results, analysis = asyncio.run(
        generate_all_mmlu_thinking(
            model=MODEL,
            max_questions=MAX_QUESTIONS,
            batch_size=BATCH_SIZE,
            max_concurrent=MAX_CONCURRENT,
            output_dir=OUTPUT_DIR,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            num_runs_per_question=NUM_RUNS_PER_QUESTION,
            max_nonthinking_accuracy=MAX_NONTHINKING_ACCURACY,
        )
    )
    # get_thinking_prompts()
    print(results[0]["all_thinking_traces"][0])
    print("-------")
    print(results[0]["all_thinking_traces"][1])

    # Display detailed statistics
    display_thinking_statistics(results, analysis)
