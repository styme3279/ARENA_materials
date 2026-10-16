import os
import json
import re
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer
import random
from datasets import load_dataset


def get_chunk_ranges(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0

    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()

        # Try to find the chunk in the full text
        chunk_start = -1

        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()

            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i : i + len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r"\s+", " ", text_window).strip()

                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break

                # If not found with window, try word by word matching
                if i == current_pos + 100:  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(
                            r"\b" + re.escape(chunk_words[0]) + r"\b",
                            full_text[j : j + len(chunk_words[0]) + 5],
                        ):
                            # Check if subsequent words match
                            match_text = full_text[j : j + len(normalized_chunk) + 30]
                            normalized_match = re.sub(r"\s+", " ", match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break

        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue

        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r"\s+", "", chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(
            r"\s+", "", full_text_from_start[: len(chunk) + 50]
        )  # Remove all whitespace

        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break

        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i

        chunk_end += 1  # Include the last character
        current_pos = chunk_end

        chunk_ranges.append((chunk_start, chunk_end))

    return chunk_ranges


def get_chunk_token_ranges(
    text: str, chunk_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    chunk_token_ranges = []

    for chunk_start, chunk_end in chunk_ranges:
        chunk_start_token = tokenizer.encode(text[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        chunk_end_token = tokenizer.encode(text[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        chunk_token_ranges.append((chunk_start_token_idx, chunk_end_token_idx))

    return chunk_token_ranges


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not boxed_starts:
        return [""]

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers if answers else [""]


def normalize_answer(answer: str, use_sympy: bool = False) -> str:
    """
    Get the final normalized and cleaned version of an answer.
    This function combines all normalization steps used in check_answer.

    Args:
        answer: The answer string to normalize
        use_sympy: Whether to use sympy to normalize the answer

    Returns:
        The normalized answer string
    """
    # First apply basic LaTeX normalization
    normalized = normalize_latex(answer)

    # Also prepare the answer for sympy if applicable
    if use_sympy:
        try:
            sympy_ready = prepare_latex_for_sympy(answer)
            if sympy_ready != normalized and len(sympy_ready) > 0:
                return sympy_ready
        except Exception:
            pass

    return normalized


def check_answer(answer: str, gt_answer: str) -> bool:
    """
    Check if the generated answer matches the ground truth answer
    after normalizing LaTeX formatting.

    Args:
        answer: The generated answer to check
        gt_answer: The ground truth answer to compare against

    Returns:
        True if the answers match after normalization, False otherwise
    """
    # Normalize both answers
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)

    # First check if normalized strings match
    if normalized_answer == normalized_gt_answer:
        return True

    # If string comparison fails, try mathematical equivalence
    try:
        return get_latex_equivalent(answer, gt_answer)
    except Exception as e:
        # If SymPy parsing fails, fall back to string comparison result
        return False


def get_latex_equivalent(answer0, answer1):
    """
    Check if two LaTeX expressions are mathematically equivalent using SymPy.

    Args:
        answer0: First LaTeX expression
        answer1: Second LaTeX expression

    Returns:
        True if expressions are mathematically equivalent, False otherwise
    """
    try:
        from sympy.parsing.latex import parse_latex
        import sympy

        # Clean up the LaTeX expressions for parsing
        answer0 = prepare_latex_for_sympy(answer0)
        answer1 = prepare_latex_for_sympy(answer1)

        # Parse the LaTeX expressions
        expr1 = parse_latex(answer0)
        expr2 = parse_latex(answer1)

        # Check if they are mathematically identical
        equals = expr1.equals(expr2)
        # print(f"First: {answer0}, Second: {answer1}: equals={equals}")
        return equals
    except Exception as e:
        # print(f"Error comparing expressions: {e}")
        return False


def prepare_latex_for_sympy(latex_str):
    """
    Prepare a LaTeX string for SymPy parsing by removing unsupported commands
    and simplifying the expression.
    """
    if not isinstance(latex_str, str):
        return str(latex_str)

    # Remove \boxed{} command
    latex_str = re.sub(r"\\boxed\{(.*?)\}", r"\1", latex_str)

    # Replace common LaTeX commands that SymPy doesn't support
    replacements = {
        r"\\dfrac": r"\\frac",
        r"\\tfrac": r"\\frac",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\div": r"/",
        r"\\left": r"",
        r"\\right": r"",
        r"\\textbf": r"",
        r"\\text": r"",
        r"\\mathrm": r"",
        r"\\!": r"",
        r",": r"",
    }

    for old, new in replacements.items():
        latex_str = re.sub(old, new, latex_str)

    return latex_str


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.

    Args:
        latex_str: The LaTeX string to normalize

    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()

    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas
    normalized = normalized.replace("{,}", "")

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized


def split_solution_keep_spacing(solution_text: str) -> List[str]:
    """
    Split a solution into chunks while preserving spacing.
    """
    # Define patterns for chunk boundaries
    sentences = split_solution_into_chunks(solution_text)
    chunk_ranges = get_chunk_ranges(solution_text, sentences)
    sentences_w_spacing = [
        solution_text[chunk_range[0] : chunk_range[1]] for chunk_range in chunk_ranges
    ]
    return sentences_w_spacing


def sanity_check_sentences(sentences_w_spacing, dir_problem, text):
    # Sanity check
    # (the chunks.json removes "\n"also omits the final sentence of the CoT)
    fp_sentences_json = os.path.join(dir_problem, "chunks.json")
    with open(fp_sentences_json, "r") as f:
        sentence_data = json.load(f)

    sentences_og = sentence_data["chunks"]

    assert len(sentences_w_spacing) == len(sentences_og) - 1
    assert (
        sentences_og[-2].strip()
        in sentences_w_spacing[-1].replace("\n", " ").replace("  ", " ").strip()
    )
    for sentence in sentences_w_spacing:
        assert sentence in text


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.

    Args:
        solution_text: The full solution text

    Returns:
        List of chunks
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into chunks
    chunks = []
    current_chunk = ""

    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # # Add the last chunk if not empty
    # if current_chunk.strip():
    #     chunks.append(current_chunk.strip())
    #     chunk_idxs.append(len(solution_text) - 1)  # Add last index

    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            # If this is the last chunk, merge with previous chunk if possible
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            # Otherwise merge with the next chunk
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
                # Don't increment i since we need to check the new merged chunk
            # If we're at the beginning and there's only one chunk, just keep it
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    # chunk_boundaries = [(chunk_idxs[i], chunk_idxs[i + 1]) for i in range(len(chunk_idxs) - 1)]
    # chunk_boundaries.append((chunk_idxs[-1], len(solution_text)))

    # if get_idxs:
    #     return chunks, chunk_boundaries
    # else:
    return chunks


def load_math_problems(
    problem_type: Optional[str] = None,
    level: Optional[str] = None,
    num_problems: Optional[int] = None,
    split: str = "train",
    include_problems: Optional[List[int]] = None,
) -> List[Tuple[int, Dict]]:
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
        indexed_problems = [
            (
                i,
                {
                    "problem": item["problem"],
                    "level": item["level"],
                    "type": item["type"],
                    "gt_solution": item["solution"],
                },
            )
            for i, item in enumerate(dataset_split)
        ]

        # Extract ground truth answers
        for i, problem in indexed_problems:
            gt_boxed_answers = extract_boxed_answers(problem["gt_solution"])
            gt_answer = gt_boxed_answers[0] if gt_boxed_answers else ""
            problem["gt_answer"] = gt_answer

        # Filter by type if specified
        if problem_type is not None:
            indexed_problems = [
                (i, problem)
                for i, problem in indexed_problems
                if problem.get("type") == problem_type
            ]

        # Filter by level if specified
        if level is not None:
            indexed_problems = [
                (i, problem) for i, problem in indexed_problems if problem.get("level") == level
            ]

        # Sample if needed
        if (
            num_problems is not None
            and include_problems is None
            and num_problems < len(indexed_problems)
        ):
            indexed_problems = random.sample(indexed_problems, num_problems)

        if level:
            print(f"Filtered to level: {level}")
        if problem_type:
            print(f"Filtered to type: {problem_type}")

        return indexed_problems
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []
