#!/usr/bin/env python
"""Generate LaTeX table of domains and subjects for paper appendix."""

from constants import (
    STEM_LOGIC_SUBJECTS,
    LIFE_SCIENCES_SUBJECTS,
    HUMANITIES_SOCIAL_SUBJECTS,
    MATH_PHYSICS_LOGIC_NAME,
    LIFE_SCIENCES_NAME,
    HUMANITIES_SOCIAL_NAME,
)


def format_subject_name(subject):
    """Convert subject identifier to readable format."""
    # Special cases
    replacements = {
        "high_school": "HS",
        "college": "College",
        "professional": "Prof.",
        "elementary": "Elem.",
        "us_foreign_policy": "US Foreign Policy",
        "high_school_us_history": "HS US History",
    }

    # Apply special replacements first
    formatted = subject
    for old, new in replacements.items():
        if old in formatted and old != "high_school_us_history":
            formatted = formatted.replace(old, new)

    # Handle the specific case
    if subject == "high_school_us_history":
        return "HS US History"

    # Replace underscores with spaces and title case
    formatted = formatted.replace("_", " ").title()

    # Fix some specific formatting
    formatted = formatted.replace("Hs ", "HS ")
    formatted = formatted.replace("Prof. ", "Prof. ")
    formatted = formatted.replace("Elem. ", "Elem. ")
    formatted = formatted.replace("Us ", "US ")

    return formatted


def generate_latex_table():
    """Generate LaTeX table with three columns for domains and subjects."""

    # Prepare the three domain lists
    domains = {
        MATH_PHYSICS_LOGIC_NAME: STEM_LOGIC_SUBJECTS,
        LIFE_SCIENCES_NAME: LIFE_SCIENCES_SUBJECTS,
        HUMANITIES_SOCIAL_NAME: HUMANITIES_SOCIAL_SUBJECTS,
    }

    # Format subject names
    formatted_domains = {}
    for domain, subjects in domains.items():
        formatted_domains[domain] = [format_subject_name(s) for s in sorted(subjects)]

    # Find the maximum number of subjects
    max_subjects = max(len(subjects) for subjects in formatted_domains.values())

    # Start building the LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{MMLU Subjects by Domain}")
    latex_lines.append("\\label{tab:mmlu_subjects}")
    latex_lines.append("\\begin{tabular}{lll}")
    latex_lines.append("\\toprule")

    # Headers (domain names in bold)
    headers = []
    for domain in domains.keys():
        # Escape ampersands in LaTeX
        domain_latex = domain.replace("&", "\\&")
        headers.append(f"\\textbf{{{domain_latex}}}")
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\midrule")

    # Add subjects row by row
    math_physics_subjects = formatted_domains[MATH_PHYSICS_LOGIC_NAME]
    life_sciences_subjects = formatted_domains[LIFE_SCIENCES_NAME]
    humanities_subjects = formatted_domains[HUMANITIES_SOCIAL_NAME]

    for i in range(max_subjects):
        row = []

        # Mathematics, Physics, and Logic column
        if i < len(math_physics_subjects):
            subject = math_physics_subjects[i]
            # Escape special LaTeX characters
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        # Life Sciences column
        if i < len(life_sciences_subjects):
            subject = life_sciences_subjects[i]
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        # Humanities/Social column
        if i < len(humanities_subjects):
            subject = humanities_subjects[i]
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        latex_lines.append(" & ".join(row) + " \\\\")

    # Close the table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def generate_compact_latex_table():
    """Generate a more compact LaTeX table using smaller font."""

    # Prepare the three domain lists
    domains = {
        MATH_PHYSICS_LOGIC_NAME: STEM_LOGIC_SUBJECTS,
        LIFE_SCIENCES_NAME: LIFE_SCIENCES_SUBJECTS,
        HUMANITIES_SOCIAL_NAME: HUMANITIES_SOCIAL_SUBJECTS,
    }

    # Format subject names (keep them shorter)
    formatted_domains = {}
    for domain, subjects in domains.items():
        formatted_domains[domain] = [format_subject_name(s) for s in sorted(subjects)]

    # Find the maximum number of subjects
    max_subjects = max(len(subjects) for subjects in formatted_domains.values())

    # Start building the LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table*}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small  % Make text smaller")
    latex_lines.append("\\caption{MMLU Subjects by Domain}")
    latex_lines.append("\\label{tab:mmlu_subjects}")
    latex_lines.append("\\begin{tabular}{p{4.5cm}p{4.5cm}p{4.5cm}}")
    latex_lines.append("\\toprule")

    # Headers (domain names in bold)
    headers = []
    for domain in domains.keys():
        # Escape ampersands in LaTeX
        domain_latex = domain.replace("&", "\\&")
        headers.append(f"\\textbf{{{domain_latex}}}")
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\midrule")

    # Add subjects row by row
    math_physics_subjects = formatted_domains[MATH_PHYSICS_LOGIC_NAME]
    life_sciences_subjects = formatted_domains[LIFE_SCIENCES_NAME]
    humanities_subjects = formatted_domains[HUMANITIES_SOCIAL_NAME]

    # Add count information
    latex_lines.append(f"\\textit{{({len(math_physics_subjects)} subjects)}} & "
                      f"\\textit{{({len(life_sciences_subjects)} subjects)}} & "
                      f"\\textit{{({len(humanities_subjects)} subjects)}} \\\\")
    latex_lines.append("\\addlinespace[0.5em]")  # Add some space

    for i in range(max_subjects):
        row = []

        # Mathematics, Physics, and Logic column
        if i < len(math_physics_subjects):
            subject = math_physics_subjects[i]
            # Escape special LaTeX characters
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        # Life Sciences column
        if i < len(life_sciences_subjects):
            subject = life_sciences_subjects[i]
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        # Humanities/Social column
        if i < len(humanities_subjects):
            subject = humanities_subjects[i]
            subject = subject.replace("&", "\\&")
            row.append(subject)
        else:
            row.append("")

        # Add some visual separation every 5 rows
        if i > 0 and i % 5 == 0:
            latex_lines.append("\\addlinespace[0.3em]")

        latex_lines.append(" & ".join(row) + " \\\\")

    # Close the table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


if __name__ == "__main__":
    print("=" * 80)
    print("STANDARD LATEX TABLE")
    print("=" * 80)
    print(generate_latex_table())
    print("\n" + "=" * 80)
    print("COMPACT VERSION (for two-column papers)")
    print("=" * 80)
    print(generate_compact_latex_table())
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  {MATH_PHYSICS_LOGIC_NAME}: {len(STEM_LOGIC_SUBJECTS)} subjects")
    print(f"  {LIFE_SCIENCES_NAME}: {len(LIFE_SCIENCES_SUBJECTS)} subjects")
    print(f"  {HUMANITIES_SOCIAL_NAME}: {len(HUMANITIES_SOCIAL_SUBJECTS)} subjects")
    print(f"  Total: {len(STEM_LOGIC_SUBJECTS) + len(LIFE_SCIENCES_SUBJECTS) + len(HUMANITIES_SOCIAL_SUBJECTS)} subjects")