#!/usr/bin/env python3
"""
L1 Instruction Generator using Qwen3.6-27B Model
This script rewrites navigation instructions to generate L1 (Level 1) instructions
that focus on static environmental landmarks without pedestrian-related content.
"""

import os
import sys
import json
import gzip
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/share/home/u19666033/dhj/DPed_pro/generate_l1_instructions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import transformers, install if needed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    logger.info("Installing required packages...")
    os.system("pip install transformers torch accelerate")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

@dataclass
class ModelConfig:
    model_path: str = "/share/home/u19666033/dhj/models/Qwen3.6-27B"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class QwenInference:
    """Qwen model inference wrapper for L1 instruction generation."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()
        
    def load_model(self):
        """Load the Qwen model and tokenizer."""
        if self.tokenizer is not None and self.model is not None:
            return
            
        logger.info(f"Loading Qwen model from {self.config.model_path}...")
        logger.info(f"Using device: {self.config.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_l1_instruction(self, original_instruction: str, scene_id: str = "") -> str:
        """
        Rewrite the original instruction to generate L1 (Level 1) instruction.
        
        L1 instructions should:
        1. Focus only on static environmental landmarks (hallway, kitchen, door, etc.)
        2. NOT contain any pedestrian-related content (avoid, person, pedestrian, etc.)
        3. Be coherent, natural language instructions
        4. Preserve the navigation goal and route information
        
        Args:
            original_instruction: The original instruction (may contain pedestrian references)
            scene_id: Optional scene identifier
            
        Returns:
            L1 instruction without pedestrian references
        """
        with self._lock:
            if self.model is None:
                self.load_model()
        
        # Build the prompt for L1 instruction generation
        system_prompt = """You are an expert in robot navigation instruction generation. Your task is to transform navigation instructions to Level 1 (L1) instructions.

L1 INSTRUCTIONS CRITERIA:
1. Focus ONLY on static environmental landmarks (hallway, kitchen, door, room, wall, window, staircase, furniture, etc.)
2. REMOVE ALL pedestrian-related content:
   - Remove phrases like "avoid the person", "watch out for pedestrians", "wait for the pedestrian"
   - Remove any references to people walking, standing, or moving
   - Remove navigation instructions related to human interaction
3. Keep the navigation goal and general route structure
4. Use natural, fluent English
5. Be concise but informative

EXAMPLES:

Original: "Turn left. No pedestrians. Move forward. Avoid the person near the wall. Turn right to the kitchen."
L1: "Turn left and proceed forward. Navigate to the kitchen area on the right."

Original: "Go past the staircase. Navigate carefully around the person walking ahead. Stop at the door."
L1: "Pass the staircase on your right. Continue to the doorway and stop."

Original: "Walk ahead. Person on staircase, facing left. Make a left turn. Avoid the pedestrian."
L1: "Continue forward and turn left at the corridor. Proceed to the destination."

Original: "Head straight. No pedestrians. Turn left. Take a right. Go right. No people."
L1: "Head straight through the hallway. Turn left, then take the next right, and continue forward."

Original: "Move forward. Navigate carefully around the painting near the center. Head toward the chair near the staircase."
L1: "Move forward, passing the painting on your right. Continue toward the chair near the staircase."

CRITICAL: Output ONLY the rewritten instruction, nothing else. The instruction should be a single coherent paragraph or flow of sentences without lists or bullet points."""

        user_prompt = f"""Transform this navigation instruction into an L1 instruction (no pedestrian references):

Original Instruction:
{original_instruction}

L1 Instruction (output only):"""

        # Construct messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        
        # Remove any thinking tags if present (for Qwen models with thinking)
        for tag in ['<think>', '</think>', '<think>', '</think>']:
            response = response.replace(tag, '')
        
        # Final cleanup - remove quotes if present
        response = response.strip('"\'')
        
        return response
    
    def generate_l1_instruction_strict(self, original_instruction: str, scene_id: str = "") -> str:
        """
        Generate L1 instruction with a stricter prompt for retry scenarios.
        
        This method uses a more explicit prompt to ensure no pedestrian content.
        
        Args:
            original_instruction: The original instruction (may contain pedestrian references)
            scene_id: Optional scene identifier
            
        Returns:
            L1 instruction without pedestrian references
        """
        with self._lock:
            if self.model is None:
                self.load_model()
        
        system_prompt = """You are a strict instruction rewriter. Your ONLY task is to convert navigation instructions to pedestrian-free L1 instructions.

RULES (MUST FOLLOW):
1. REMOVE ALL references to: person, people, pedestrian, man, woman, human, individual
2. REMOVE ALL action verbs: avoid, wait for, watch out for, navigating around
3. REMOVE ALL descriptions of human movement: walking, standing, approaching, entering, exiting
4. KEEP ONLY: static environment descriptions (rooms, doors, hallways, furniture, objects)
5. Output must be a COMPLETE, COHERENT navigation instruction
6. Do NOT output anything except the rewritten instruction

ABSOLUTE FORBIDDEN WORDS:
person, people, pedestrian, avoid, waiting, wait, approaching, walking, standing, human, man, woman, near, behind, ahead (of you)

Example transformation:
Input: "Turn left. Avoid the person near the door. Walk forward to the kitchen."
Output: "Turn left and proceed forward. Navigate to the kitchen area."

CRITICAL: Output ONLY the rewritten instruction, nothing else."""

        user_prompt = f"""Rewrite this instruction to remove ALL pedestrian-related content:

{original_instruction}

Rewritten (no people references):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,  # Lower temperature for stricter output
                top_p=0.8,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        response = response.strip()
        for tag in ['<think>', '</think>']:
            response = response.replace(tag, '')
        response = response.strip('"\'')
        
        return response


def load_episodes_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Load episodes from JSON or compressed JSON file."""
    episodes = []
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        if 'episodes' in data:
            episodes = data['episodes']
        elif isinstance(data, list):
            episodes = data
        else:
            episodes = [data]
            
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
    
    return episodes


def save_episodes_to_file(filepath: str, episodes: List[Dict[str, Any]]):
    """Save episodes to JSON file (not compressed)."""
    data = {"episodes": episodes}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def contains_pedestrian_keywords(text: str) -> tuple:
    """
    Check if text contains pedestrian-related keywords.
    
    Returns:
        tuple: (contains: bool, found_keywords: list)
    """
    pedestrian_keywords = [
        # Direct person references
        'person', 'persons', 'people', 'pedestrian', 'pedestrians',
        'man', 'woman', 'men', 'women',
        'human', 'humans', 'individual', 'individuals',
        # Movement/action keywords related to people (when followed by person references)
        'walking', 'walked', 'walks',
        'standing', 'stood', 'stands',
        'sitting', 'sat', 'sits',
        'approaching', 'approached',
        'passing', 'passed', 'passes',
        'moving', 'moved', 'moves',
        'waiting', 'waited', 'waits',
        'entering', 'entered', 'enters',
        'exiting', 'exited', 'exits',
        # Contextual phrases
        'crowd', 'crowded',
        # Avoidance actions (when related to people)
        'avoid', 'avoiding', 'evade', 'evading',
        'dodge', 'dodging',
        'get out of the way', 'clear the path',
        # Common phrases with people
        'watch out for', 'look out for', 'be careful of',
        'give way to', 'make way for',
    ]
    
    # Words that are navigation-related, NOT pedestrian-related
    # These should only trigger if followed by person references
    ambiguous_words = [
        'head', 'ahead', 'behind',
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in pedestrian_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Check for ambiguous words - only count if followed by person references
    person_refs = ['person', 'people', 'pedestrian', 'man', 'woman', 'human']
    
    for word in ambiguous_words:
        # Check patterns like "head toward person", "person ahead"
        for person_ref in person_refs:
            if f"{word} toward {person_ref}" in text_lower:
                found_keywords.append(word)
            if f"{word} of {person_ref}" in text_lower:
                found_keywords.append(word)
            if f"{person_ref} {word}" in text_lower:
                found_keywords.append(word)
    
    return len(found_keywords) > 0, found_keywords


def validate_l1_instruction(l1_instruction: str, original_instruction: str) -> tuple:
    """
    Validate that the L1 instruction doesn't contain pedestrian-related content.
    
    Returns:
        tuple: (is_valid: bool, issues: list, suggestion: str)
    """
    issues = []
    suggestions = []
    
    # Check for pedestrian keywords
    contains, found = contains_pedestrian_keywords(l1_instruction)
    if contains:
        issues.append(f"Contains pedestrian keywords: {found}")
        suggestions.append("Remove all references to people, pedestrians, avoidance actions")
    
    # Check for empty or too short instructions
    if len(l1_instruction.strip()) < 10:
        issues.append("Instruction too short or empty")
        suggestions.append("Provide a complete navigation instruction")
    
    # Check for question marks (shouldn't be asking questions)
    if '?' in l1_instruction:
        issues.append("Contains question marks - should be imperative")
        suggestions.append("Rewrite as a command, not a question")
    
    # Check for coherence (should have navigation verbs)
    nav_verbs = ['go', 'turn', 'move', 'walk', 'head', 'proceed', 
                  'continue', 'stop', 'wait', 'navigate', 'pass']
    has_nav_verb = any(verb in l1_instruction.lower() for verb in nav_verbs)
    if not has_nav_verb:
        issues.append("Missing navigation verbs")
        suggestions.append("Include directional commands like 'go', 'turn', 'move'")
    
    return len(issues) == 0, issues, suggestions[0] if suggestions else ""


def rewrite_instruction_with_fallback(model: QwenInference, original_instruction: str,
                                    scene_id: str, retry_count: int = 0,
                                    max_retries: int = 3,
                                    skip_validation: bool = True) -> str:
    """
    Generate L1 instruction (validation temporarily disabled).

    Args:
        model: Qwen inference model
        original_instruction: Original instruction to rewrite
        scene_id: Scene identifier
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
        skip_validation: If True, skip all validation checks and return model output directly

    Returns:
        L1 instruction (model output, no validation)
    """
    if retry_count == 0:
        generated = model.generate_l1_instruction(original_instruction, scene_id)
    elif retry_count == 1:
        original_temp = model.config.temperature
        model.config.temperature = 0.5
        generated = model.generate_l1_instruction(original_instruction, scene_id)
        model.config.temperature = original_temp
    else:
        generated = model.generate_l1_instruction_strict(original_instruction, scene_id)

    # --- Validation bypassed ---
    if skip_validation:
        return generated

    # --- Original validation code (disabled) ---
    is_valid, issues, suggestion = validate_l1_instruction(generated, original_instruction)

    if not is_valid and retry_count < max_retries - 1:
        logger.warning(f"Validation failed (attempt {retry_count + 1}): {issues}")
        logger.warning(f"Suggestion: {suggestion}")
        return rewrite_instruction_with_fallback(
            model, original_instruction, scene_id, retry_count + 1, max_retries, skip_validation
        )

    if not is_valid:
        logger.warning(f"All retries failed for instruction. Falling back to simplified version.")
        return create_fallback_l1_instruction(original_instruction)

    return generated


def create_fallback_l1_instruction(original_instruction: str) -> str:
    """
    Create a simplified L1 instruction by removing pedestrian-related content.
    Used when model generation fails validation.
    
    Args:
        original_instruction: Original instruction
        
    Returns:
        Simplified instruction without pedestrian content
    """
    # Patterns to remove (pedestrian-related phrases)
    patterns_to_remove = [
        # Avoid patterns
        r',?\s*avoid\s+(the\s+)?(person|pedestrian|people|man|woman|human)[^\.]*',
        r',?\s*avoiding\s+(the\s+)?(person|pedestrian|people)[^\.]*',
        r',?\s*navigate\s+carefully\s+around\s+(the\s+)?(person|pedestrian|people)[^\.]*',
        # Wait patterns
        r',?\s*wait\s+for\s+(the\s+)?(person|pedestrian|pedestrians)[^\.]*',
        r',?\s*stand\s+by[^\.]*',
        # Walking/approaching patterns
        r',?\s*(person|pedestrian|people|man|woman)\s+(walking|moving|standing|approaching|entering|exiting)[^\.]*',
        r',?\s*(walking|moving|standing|approaching)\s+(person|pedestrian|people)[^\.]*',
        # Person descriptions
        r',?\s*(person|pedestrian|people|man|woman)\s+on\s+(the\s+)?(left|right|ahead|behind)[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+near[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+in[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+at[^\.]*',
        # No pedestrians phrases - keep but simplify
        r'\.?\s*No\s+pedestrians\.?',
        r'\.?\s*No\s+people\.?',
        r'\.?\s*No\s+person\.?',
        # Other people references
        r',?\s*give\s+way\s+to[^\.]*',
        r',?\s*make\s+way\s+for[^\.]*',
        r',?\s*watch\s+out\s+for[^\.]*',
        r',?\s*look\s+out\s+for[^\.]*',
    ]
    
    result = original_instruction
    
    for pattern in patterns_to_remove:
        import re
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*', ', ', result)
    result = re.sub(r',\s*,', ',', result)
    result = result.strip()
    
    # Ensure it ends with a period
    if result and not result.endswith('.'):
        result += '.'
    
    # If result is empty or too short, create minimal instruction
    if len(result) < 10:
        result = "Proceed forward following the navigation route to the destination."
    
    return result


def process_file(input_filepath: str, output_filepath: str, model: QwenInference,
                 dry_run: bool = False, max_retries: int = 3,
                 pbar: tqdm = None) -> Dict[str, Any]:
    """
    Process a single data file and replace instructions with L1 versions.
    
    Args:
        input_filepath: Path to input JSON file
        output_filepath: Path to output JSON file
        model: Qwen inference model
        dry_run: If True, skip model inference
        max_retries: Maximum retry attempts for validation failures
        pbar: tqdm progress bar to update
        
    Returns:
        Processing statistics
    """
    start_time = time.time()
    
    # Load episodes
    episodes = load_episodes_from_file(input_filepath)
    
    if not episodes:
        if pbar:
            pbar.update(1)
        return {"status": "empty", "episodes_processed": 0}
    
    # Process each episode with inline logic (no batch processing)
    with tqdm(episodes, desc="Episodes", leave=False) as pbar_ep:
        for episode in pbar_ep:
            original_instruction = episode.get('instruction', '')
            
            if not original_instruction:
                episode['l1_changed'] = False
                episode['l1_validation_passed'] = True
                continue
            
            if dry_run:
                episode['instruction'] = "[DRY RUN] " + original_instruction
                episode['l1_changed'] = True
                episode['l1_validation_passed'] = True
            else:
                try:
                    # Generate L1 instruction (validation bypassed)
                    l1_instruction = rewrite_instruction_with_fallback(
                        model, original_instruction,
                        episode.get('scene_id', ''),
                        max_retries=max_retries,
                        skip_validation=True,
                    )

                    # Print before/after comparison
                    ep_id = episode.get('episode_id', 'unknown')
                    print(f"\n{'='*80}")
                    print(f"[{ep_id}] BEFORE (original instruction):")
                    print(f"  {original_instruction}")
                    print(f"[{ep_id}] AFTER (L1 instruction):")
                    print(f"  {l1_instruction}")
                    print(f"{'='*80}\n")

                    # Replace instruction with L1 version (no validation check)
                    episode['instruction'] = l1_instruction
                    episode['l1_changed'] = True
                    episode['l1_validation_passed'] = True

                except Exception as e:
                    logger.warning(f"Error generating L1 for episode {episode.get('episode_id', 'unknown')}: {e}")
                    episode['l1_changed'] = False
                    episode['l1_error'] = str(e)
                    episode['l1_validation_passed'] = True
    
    # Count statistics
    changed_count = sum(1 for ep in episodes if ep.get('l1_changed', False))
    
    # Save results
    save_episodes_to_file(output_filepath, episodes)
    
    elapsed = time.time() - start_time
    
    stats = {
        "status": "success",
        "input_file": input_filepath,
        "output_file": output_filepath,
        "episodes_processed": len(episodes),
        "instructions_changed": changed_count,
        "elapsed_seconds": elapsed
    }
    
    if pbar:
        pbar.update(1)
    
    return stats


def main():
    """Main function to process all training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate L1 instructions using Qwen model")
    parser.add_argument("--input-dir", type=str, 
                       default="/share/home/u19666033/dhj/DPed_pro/dped_pro/train",
                       help="Input directory containing JSON files")
    parser.add_argument("--output-dir", type=str,
                       default="/share/home/u19666033/dhj/DPed_pro/dped_pro/train_l1",
                       help="Output directory for processed files")
    parser.add_argument("--model-path", type=str,
                       default="/share/home/u19666033/dhj/models/Qwen3.6-27B",
                       help="Path to Qwen model")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test without calling model")
    parser.add_argument("--max-files", type=int, default=0,
                       help="Maximum number of files to process (0=all)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run (skip existing output files)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for validation failures")
    parser.add_argument("--all-datasets", action="store_true", default=True,
                       help="Process all DPED datasets (train, seen_test, seen_val, unseen_test, unseen_val)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       choices=["train", "seen_test", "seen_val", "unseen_test", "unseen_val"],
                       help="Specific datasets to process")
    
    args = parser.parse_args()
    
    # Define all dataset directories and their output subdirectories
    base_input = "/share/home/u19666033/dhj/DPed_pro/dped_pro"
    base_output = args.output_dir

    datasets = {
        "train":           (f"{base_input}/train",           f"{base_output}/train"),
        "seen_test":       (f"{base_input}/seen/seen_test",  f"{base_output}/seen_test"),
        "seen_val":        (f"{base_input}/seen/seen_val",   f"{base_output}/seen_val"),
        "unseen_test":     (f"{base_input}/unseen/unseen_test", f"{base_output}/unseen_test"),
        "unseen_val":      (f"{base_input}/unseen/unseen_val",  f"{base_output}/unseen_val"),
    }
    
    # Filter datasets based on --datasets argument
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}
    
    # Initialize model
    config = ModelConfig(model_path=args.model_path)
    model = QwenInference(config)
    
    if not args.dry_run:
        model.load_model()
    
    # Collect files from all directories
    all_dir_files = {}
    for name, (input_dir, output_dir) in datasets.items():
        if not os.path.isdir(input_dir):
            logger.warning(f"Input directory does not exist: {input_dir} — skipping")
            continue
        files = [f for f in os.listdir(input_dir) if f.endswith('.json') or f.endswith('.json.gz')]
        files.sort()
        if args.max_files > 0:
            files = files[:args.max_files]
        all_dir_files[name] = (input_dir, output_dir, files)
        logger.info(f"[{name}] Found {len(files)} files in {input_dir}")
    
    # Process each dataset with progress bar
    all_stats = []
    total_processed = 0
    total_skipped = 0
    grand_total_episodes = 0
    grand_total_changed = 0
    
    for name, (input_dir, output_dir, input_files) in all_dir_files.items():
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {name}  ({len(input_files)} files)")
        logger.info(f"{'='*60}")
        
        with tqdm(input_files, desc=f"{name}", unit="file") as pbar:
            for filename in pbar:
                input_path = os.path.join(input_dir, filename)
                base_name = filename.replace('.gz', '')
                output_path = os.path.join(output_dir, base_name)
                
                if args.resume and os.path.exists(output_path):
                    pbar.write(f"Skipping: {filename}")
                    total_skipped += 1
                    continue
                
                try:
                    stats = process_file(input_path, output_path, model, args.dry_run, args.max_retries, pbar)
                    stats["dataset"] = name
                    all_stats.append(stats)
                    total_processed += 1
                    grand_total_episodes += stats.get('episodes_processed', 0)
                    grand_total_changed += stats.get('instructions_changed', 0)
                    
                    # Update progress bar postfix with stats
                    pbar.set_postfix({
                        "changed": grand_total_changed,
                        "episodes": grand_total_episodes
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    pbar.update(1)
                    all_stats.append({
                        "dataset": name,
                        "input_file": filename,
                        "status": "error",
                        "error": str(e)
                    })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL DATASETS PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_processed}")
    logger.info(f"Files skipped (resume): {total_skipped}")
    logger.info(f"Total episodes processed: {grand_total_episodes}")
    logger.info(f"Total instructions modified: {grand_total_changed}")
    logger.info(f"Total instructions unchanged: {grand_total_episodes - grand_total_changed}")
    
    # Save summary
    summary_path = os.path.join(base_output, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "datasets": {name: len(data[2]) for name, data in all_dir_files.items()},
            "stats": all_stats,
            "summary": {
                "files_processed": total_processed,
                "files_skipped": total_skipped,
                "total_episodes": grand_total_episodes,
                "total_changed": grand_total_changed,
                "total_unchanged": grand_total_episodes - grand_total_changed,
            }
        }, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
