import re
from typing import List, Tuple
from pkld import pkld


def clean_python_string_literal(text: str) -> str:
    """
    Clean a Python string literal that may contain quotes, escaped characters, etc.
    """
    # Remove outer parentheses if present
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    # Handle multiline Python string concatenation
    # Remove quotes and string concatenation operators
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading/trailing quotes and concatenation
        line = re.sub(r'^["\']', "", line)  # Remove leading quote
        line = re.sub(r'["\']$', "", line)  # Remove trailing quote
        line = re.sub(r'^["\']', "", line)  # Remove any remaining leading quote
        line = re.sub(r'["\']$', "", line)  # Remove any remaining trailing quote

        cleaned_lines.append(line)

    # Join all lines with spaces
    result = " ".join(cleaned_lines)

    # Replace escaped characters
    result = result.replace("\\n", " ")
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace("\\\\", "\\")

    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


@pkld(store="both", overwrite=False)
def string_to_sentences(text: str, drop_post_think=False) -> tuple[List[str], List[int]]:
    """
    Convert a string into a list of sentences and their starting positions.

    Handles:
    - Standard sentence endings (. ! ?)
    - Headers and titles on separate lines
    - Quotes and parentheses
    - Newlines and spacing
    - Common abbreviations
    - Python string literals

    Args:
        text (str): Input text to split into sentences

    Returns:
        tuple[List[str], List[int]]: List of cleaned sentences and their starting character positions
    """
    if not text or not isinstance(text, str):
        return [], []

    # Store original text before any processing
    original_text = text

    # First, split on newlines to handle headers/titles
    line_segments = []
    current_pos = 0

    # Split by newlines but keep track of positions
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip():  # Only process non-empty lines
            # Find the actual position of this line in the original text
            line_pos = original_text.find(line, current_pos)
            if line_pos == -1:
                line_pos = current_pos

            # Check if this line is a header/title
            # Headers are often: standalone lines, lines with **, lines ending with colon, short lines without ending punctuation
            line_stripped = line.strip()
            is_header = False

            # Check for markdown-style headers or titles
            if (
                (line_stripped.startswith("**") and line_stripped.endswith("**"))
                or (line_stripped.startswith("#"))
                or (line_stripped.endswith(":") and len(line_stripped) < 100)
                or (
                    len(line_stripped) < 80
                    and not any(line_stripped.endswith(p) for p in [".", "!", "?", '"', "'"])
                )
            ):
                is_header = True

            # Check if it's the only content on the line (before and after are empty or newlines)
            # This catches standalone titles
            if drop_post_think:
                if i > 0 and i < len(lines) - 1:
                    if (not lines[i - 1].strip() or lines[i - 1].strip().endswith("</think>")) and (
                        not lines[i + 1].strip() or lines[i + 1].strip().startswith("<")
                    ):
                        is_header = True

            line_segments.append({"text": line, "position": line_pos, "is_header": is_header})

        current_pos += len(line) + 1  # +1 for the newline character

    # Now process each segment
    all_sentences = []
    all_positions = []

    for segment in line_segments:
        if segment["is_header"]:
            # Treat the entire line as a single sentence
            sentence = segment["text"].strip()
            if sentence and len(sentence) >= 4:  # Filter very short segments
                all_sentences.append(sentence)
                all_positions.append(segment["position"])
        else:
            # Apply the existing sentence splitting logic to this segment
            segment_text = segment["text"]
            segment_start_pos = segment["position"]

            # Process this segment with the original logic
            segment_sentences, segment_positions = process_text_segment(
                segment_text, segment_start_pos
            )
            all_sentences.extend(segment_sentences)
            all_positions.extend(segment_positions)

    return all_sentences, all_positions


def process_text_segment(text: str, start_position: int = 0) -> tuple[List[str], List[int]]:
    """
    Process a text segment using the original sentence splitting logic.

    Args:
        text: The text segment to process
        start_position: The starting position of this segment in the original text

    Returns:
        tuple[List[str], List[int]]: Sentences and their positions
    """
    if not text or not isinstance(text, str):
        return [], []

    # Clean Python string literal formatting if present
    text = clean_python_string_literal(text)

    # Store original text for position finding
    original_text = text

    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = {
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "vs.",
        "e.g.",
        "i.e.",
        "cf.",
        "al.",
        "Inc.",
        "Corp.",
        "Ltd.",
        "Co.",
        "U.S.",
        "U.K.",
        # Only include degree abbreviations with periods in the middle
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "B.S.",
        "M.S.",
        "M.Ed.",
        "M.S.Ed.",
        "J.D.",
        "LL.B.",
        "LL.M.",
        "M.B.A.",
    }

    # Degree abbreviations that CAN end sentences when followed by capital letter/number
    degree_abbreviations = {
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "B.S.",
        "M.S.",
        "Ed.",
        "M.Ed.",
        "M.S.Ed.",
        "J.D.",
        "LL.B.",
        "LL.M.",
        "M.B.A.",
        "PhD.",
        "MD.",
        "BA.",
        "MA.",
        "BS.",
        "MS.",
        "MBA.",
        "JD.",
        "LLB.",
        "LLM.",
    }

    # Temporarily replace abbreviations to protect them
    protected_text = text
    abbrev_placeholders = {}

    for i, abbrev in enumerate(abbreviations):
        # Use word boundary to avoid matching abbreviations inside words
        if abbrev in protected_text:
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            if re.search(pattern, protected_text):
                placeholder = f"__ABBREV_{i}__"
                abbrev_placeholders[placeholder] = abbrev
                protected_text = re.sub(pattern, placeholder, protected_text)

    # Split on sentence endings followed by whitespace and capital letter or quote
    sentence_pattern = r'([.!?])\s*(?=[A-Z]|["\']\s*[A-Z]|\d|$)'

    # Split the text
    parts = re.split(sentence_pattern, protected_text)

    sentences = []
    current_sentence = ""

    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            # This part ends with punctuation
            current_sentence += parts[i] + parts[i + 1]
            sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 2
        else:
            # This part doesn't end with punctuation
            current_sentence += parts[i]
            i += 1

    # Add any remaining text as a sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Restore abbreviations
    restored_sentences = []
    for sentence in sentences:
        for placeholder, abbrev in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, abbrev)
        restored_sentences.append(sentence)

    # Second pass: check for abbreviations that should end sentences
    final_sentences = []
    for sentence in restored_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if this sentence contains a degree abbreviation followed by space + capital/number
        split_positions = []

        for abbrev in degree_abbreviations:
            pattern = re.escape(abbrev) + r"\s+(?=[A-Z0-9])"
            for match in re.finditer(pattern, sentence):
                split_positions.append(match.start() + len(abbrev))

        if split_positions:
            split_positions.sort()
            last_pos = 0

            for pos in split_positions:
                part = sentence[last_pos:pos].strip()
                if part:
                    final_sentences.append(part)
                last_pos = pos

            if last_pos < len(sentence):
                remaining = sentence[last_pos:].strip()
                if remaining:
                    final_sentences.append(remaining)
        else:
            final_sentences.append(sentence)

    # Find positions in the original text segment
    final_sentences_clean = []
    final_positions = []

    for sentence in final_sentences:
        sentence = sentence.strip()
        if len(sentence) < 4:
            continue

        # Search from the last found position to avoid duplicates
        search_start = 0
        if final_positions:
            # Adjust for the fact we're searching within the segment
            prev_pos_in_segment = final_positions[-1] - start_position
            search_start = prev_pos_in_segment + len(final_sentences_clean[-1])

        # Try to find the sentence in the text segment
        pos = original_text.find(sentence, search_start)

        if pos == -1:
            sentence_core = sentence.strip()
            pos = original_text.find(sentence_core, search_start)

        if pos == -1:
            normalized_sentence = " ".join(sentence.split())
            normalized_original = " ".join(original_text.split())
            norm_pos = normalized_original.find(normalized_sentence, search_start)
            if norm_pos != -1:
                pos = norm_pos

        if pos != -1:
            final_sentences_clean.append(sentence)
            # Add the start_position offset to get the position in the full text
            final_positions.append(start_position + pos)

    return final_sentences_clean, final_positions


def split_into_paragraphs(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text into paragraphs with special handling for lists and return their positions.

    Rules:
    1. Split on double newlines (\n\n)
    2. Merge consecutive list items (starting with -, *, or numbers)
    3. Filter out paragraphs < 64 characters
    4. Paragraph positions must align with sentence positions

    Returns:
        Tuple[List[str], List[int]]: List of paragraphs and their starting character positions in the original text
    """
    if not text or not isinstance(text, str):
        return [], []

    # First get sentence positions to ensure alignment

    sentences, sentence_positions = string_to_sentences(text)

    # Create a mapping of position to sentence for quick lookup
    position_to_sentence = {pos: sent for pos, sent in zip(sentence_positions, sentences)}

    # Store original text
    original_text = text

    # First pass: identify all paragraphs and whether they are lists
    raw_paragraphs = text.split("\n\n")
    paragraph_info = []
    text_position = 0

    for para_idx, para in enumerate(raw_paragraphs):
        # Find the position of this paragraph in original text
        if para_idx > 0:
            # Account for the \n\n we split on
            text_position = original_text.find(para, text_position)
            if text_position == -1:
                # Fallback: try to find without exact match
                text_position = len("\n\n".join(raw_paragraphs[:para_idx])) + 2 * para_idx

        para_start_position = text_position

        # Find the closest sentence position at or before this paragraph start
        closest_sentence_pos = None
        for sent_pos in sorted(sentence_positions):
            if sent_pos <= para_start_position:
                closest_sentence_pos = sent_pos
            else:
                break

        # If we can't find a sentence position, use the paragraph position
        if closest_sentence_pos is None:
            closest_sentence_pos = para_start_position

        # Check if this paragraph is a list or contains list items
        lines = para.strip().split("\n")
        is_list_para = False
        has_intro_line = False

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (
                re.match(r"^[-*•]\s+", stripped)
                or re.match(r"^\d+\.\s+", stripped)
                or re.match(r"^[a-zA-Z]\.\s+", stripped)
            ):
                is_list_para = True
                if line_idx > 0:
                    has_intro_line = True
                break

        # Also check if it starts with a header like "Possible actions:" followed by a list
        first_line = lines[0].strip() if lines else ""
        if first_line.endswith(":") and len(lines) > 1:
            # Check if next lines are list items
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and (
                    re.match(r"^[-*•]\s+", stripped)
                    or re.match(r"^\d+\.\s+", stripped)
                    or re.match(r"^[a-zA-Z]\.\s+", stripped)
                ):
                    is_list_para = True
                    has_intro_line = True
                    break

        paragraph_info.append(
            {
                "text": para,
                "position": closest_sentence_pos,  # Use sentence-aligned position
                "is_list": is_list_para,
                "has_intro": has_intro_line,
                "lines": lines,
            }
        )

        text_position += len(para)

    # Second pass: merge paragraphs, including intro sentences before lists
    merged_paragraphs = []
    merged_positions = []
    i = 0

    while i < len(paragraph_info):
        current_info = paragraph_info[i]

        # Check if next paragraph is a list without intro
        if (
            i + 1 < len(paragraph_info)
            and not current_info["is_list"]
            and paragraph_info[i + 1]["is_list"]
            and not paragraph_info[i + 1]["has_intro"]
        ):

            # This might be an intro paragraph for the list
            # Check if current paragraph ends with : or contains introducing language
            current_text = current_info["text"].strip()
            if (
                current_text.endswith(":")
                or "following" in current_text.lower()
                or "each" in current_text.lower()
                or "context" in current_text.lower()
            ):

                # Merge this paragraph with following list paragraphs
                merged_parts = [current_info["text"]]
                start_position = current_info["position"]

                # Collect all following list paragraphs
                j = i + 1
                while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
                    merged_parts.append(paragraph_info[j]["text"])
                    j += 1

                merged_text = "\n\n".join(merged_parts)
                if len(merged_text) >= 64:
                    merged_paragraphs.append(merged_text)
                    merged_positions.append(start_position)

                i = j
                continue

        if current_info["is_list"]:
            # Start collecting consecutive list paragraphs
            list_parts = [current_info["text"]]
            list_position = current_info["position"]

            # Look ahead for more list paragraphs
            j = i + 1
            while j < len(paragraph_info) and paragraph_info[j]["is_list"]:
                list_parts.append(paragraph_info[j]["text"])
                j += 1

            # Merge all list parts
            merged_text = "\n\n".join(list_parts)
            if len(merged_text) >= 64:
                merged_paragraphs.append(merged_text)
                merged_positions.append(list_position)

            i = j
        else:
            # Regular paragraph
            if len(current_info["text"]) >= 64:
                merged_paragraphs.append(current_info["text"])
                merged_positions.append(current_info["position"])
            i += 1

    return merged_paragraphs, merged_positions


def split_into_paragraphs_safe(text: str, allow_0: bool = False) -> Tuple[List[str], List[int]]:
    _, positions = string_to_sentences(text)
    paragraphs, paragraph_positions = split_into_paragraphs(text)
    positions_set = set(positions)

    clean_paragraphs = []
    clean_paragraph_positions = []
    for pos, para in zip(paragraph_positions, paragraphs):
        if pos == 0 and 1 in positions_set:
            clean_paragraphs.append(para[1:])
            clean_paragraph_positions.append(pos + 1)
            continue
        else:
            clean_paragraphs.append(para)
            clean_paragraph_positions.append(pos)
        # if allow_0:
        #     if pos == 0:
        #         continue
        assert (
            pos in positions_set
        ), f"Bad paragraph misaligned with sentences: {pos=}, {paragraph_positions=}, {positions=} | {para=}"
    return clean_paragraphs, clean_paragraph_positions
