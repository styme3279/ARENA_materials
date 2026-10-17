#!/usr/bin/env python3
"""
Script to download MMLU and MMLU-Pro datasets from Hugging Face.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import argparse
from tqdm import tqdm


def download_mmlu(output_dir="data/mmlu"):
    """Download the MMLU dataset from Hugging Face."""
    print("Downloading MMLU dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available subjects
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]
    
    all_data = {}
    
    # Download each subject
    for subject in tqdm(subjects, desc="Downloading MMLU subjects"):
        try:
            # Load all splits for each subject
            dataset = load_dataset("cais/mmlu", subject)
            
            subject_data = {}
            for split in dataset.keys():
                subject_data[split] = []
                for item in dataset[split]:
                    subject_data[split].append({
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"]
                    })
            
            all_data[subject] = subject_data
            
            # Save individual subject file
            subject_file = os.path.join(output_dir, f"{subject}.json")
            with open(subject_file, 'w', encoding='utf-8') as f:
                json.dump(subject_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error downloading {subject}: {e}")
    
    # Save all data in one file
    all_data_file = os.path.join(output_dir, "all_subjects.json")
    with open(all_data_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"MMLU dataset saved to {output_dir}")
    print(f"Total subjects downloaded: {len(all_data)}")
    
    # Print statistics
    total_questions = 0
    for subject, data in all_data.items():
        for split, items in data.items():
            total_questions += len(items)
    print(f"Total questions: {total_questions}")
    
    return all_data


def download_mmlu_pro(output_dir="data/mmlu_pro"):
    """Download the MMLU-Pro dataset from Hugging Face."""
    print("\nDownloading MMLU-Pro dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the MMLU-Pro dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        
        all_data = {}
        
        # Process each split
        for split in dataset.keys():
            print(f"Processing {split} split...")
            split_data = []
            
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                question_data = {
                    "question_id": item.get("question_id"),
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "answer": item.get("answer"),
                    "answer_index": item.get("answer_index"),
                    "cot_content": item.get("cot_content"),
                    "category": item.get("category"),
                    "src": item.get("src")
                }
                split_data.append(question_data)
            
            all_data[split] = split_data
            
            # Save split file
            split_file = os.path.join(output_dir, f"{split}.json")
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        # Save all data in one file
        all_data_file = os.path.join(output_dir, "all_splits.json")
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"MMLU-Pro dataset saved to {output_dir}")
        
        # Print statistics
        for split, data in all_data.items():
            print(f"{split}: {len(data)} questions")
        
        # Category statistics
        if "test" in all_data:
            categories = {}
            for item in all_data["test"]:
                cat = item.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            
            print("\nCategory distribution in test set:")
            for cat, count in sorted(categories.items()):
                print(f"  {cat}: {count} questions")
        
        return all_data
        
    except Exception as e:
        print(f"Error downloading MMLU-Pro: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download MMLU and MMLU-Pro datasets")
    parser.add_argument("--mmlu-dir", type=str, default="data/mmlu",
                        help="Output directory for MMLU dataset")
    parser.add_argument("--mmlu-pro-dir", type=str, default="data/mmlu_pro",
                        help="Output directory for MMLU-Pro dataset")
    parser.add_argument("--dataset", type=str, choices=["mmlu", "mmlu-pro", "both"],
                        default="both", help="Which dataset(s) to download")
    
    args = parser.parse_args()
    
    # Install required packages if not already installed
    try:
        import datasets
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "tqdm"])
        print("Packages installed. Please run the script again.")
        return
    
    if args.dataset in ["mmlu", "both"]:
        mmlu_data = download_mmlu(args.mmlu_dir)
        if mmlu_data:
            print("\n✓ MMLU dataset downloaded successfully!")
    
    if args.dataset in ["mmlu-pro", "both"]:
        mmlu_pro_data = download_mmlu_pro(args.mmlu_pro_dir)
        if mmlu_pro_data:
            print("\n✓ MMLU-Pro dataset downloaded successfully!")
    
    print("\nDownload complete!")
    print("\nDataset locations:")
    if args.dataset in ["mmlu", "both"]:
        print(f"  MMLU: {args.mmlu_dir}")
    if args.dataset in ["mmlu-pro", "both"]:
        print(f"  MMLU-Pro: {args.mmlu_pro_dir}")


if __name__ == "__main__":
    main()
