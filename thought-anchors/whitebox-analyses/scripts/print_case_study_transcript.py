import os
import sys
import json
import re
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_to_latex(text):
    """Convert text to LaTeX format, handling subscripts and special characters."""
    # Convert Unicode subscripts to LaTeX subscripts
    text = re.sub(r'₁₆', r'$_{16}$', text)
    text = re.sub(r'₂', r'$_2$', text)
    
    # Escape LaTeX special characters (except $ which we're using for math)
    latex_chars = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_', 
                   '{': r'\{', '}': r'\}', '~': r'\textasciitilde', 
                   '^': r'\textasciicircum'}
    
    # First protect our math mode markers
    text = text.replace('$_{16}$', '<<<SUB16>>>')
    text = text.replace('$_2$', '<<<SUB2>>>')
    
    # Escape special characters
    for char, escaped in latex_chars.items():
        text = text.replace(char, escaped)
    
    # Restore math mode markers
    text = text.replace('<<<SUB16>>>', '$_{16}$')
    text = text.replace('<<<SUB2>>>', '$_2$')
    
    # Handle multiplication symbol
    text = text.replace('×', r'$\times$')
    
    return text

def print_transcript(args):
    """Print transcript in specified format.
    
    Args:
        args: Command line arguments
    """
    # Construct input file path
    fp_in = Path(args.base_dir) / args.model_name / args.temperature_folder / args.solution_type / f"problem_{args.problem_num}" / args.input_file
    
    if not fp_in.exists():
        print(f"Error: Input file not found: {fp_in}")
        return
    
    with open(fp_in, "r") as f:
        chunks_data = json.load(f)
    
    # Generate output
    if args.output_format == "latex":
        for chunk in chunks_data:
            if chunk["function_tags"] or args.include_empty_tags:  # Check based on flag
                if chunk["function_tags"]:
                    function_tag = chunk["function_tags"][0]
                else:
                    function_tag = "none"
                chunk_text = convert_to_latex(chunk["chunk"])
                print(f"\\item (\\texttt{{{function_tag}}}): {chunk_text}")
    else:  # plain text format
        for chunk in chunks_data:
            if chunk["function_tags"] or args.include_empty_tags:
                if chunk["function_tags"]:
                    function_tag = chunk["function_tags"][0]
                else:
                    function_tag = "none"
                chunk_text = chunk["chunk"]
                print(f"- ({function_tag}): {chunk_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print case study transcript in LaTeX or plain text format")
    
    # File path configuration
    parser.add_argument("--base-dir", type=str, default="math-rollouts", help="Base directory for rollouts")
    parser.add_argument("--model-name", type=str, default="deepseek-r1-distill-qwen-14b", help="Model name")
    parser.add_argument("--temperature-folder", type=str, default="temperature_0.6_top_p_0.95", help="Temperature folder name")
    parser.add_argument("--solution-type", type=str, default="correct_base_solution", 
                       choices=["correct_base_solution", "incorrect_base_solution"], 
                       help="Solution type (correct or incorrect)")
    parser.add_argument("--problem-num", type=int, default=4682, help="Problem number")
    parser.add_argument("--input-file", type=str, default="chunks_labeled.json", help="Input JSON file name")
    
    # Output configuration
    parser.add_argument("--output-format", type=str, default="latex", 
                       choices=["latex", "plain"], 
                       help="Output format (latex or plain text)")
    parser.add_argument("--include-empty-tags", action="store_true", 
                       help="Include chunks with empty function tags")
    
    args = parser.parse_args()
    
    print_transcript(args)