#!/usr/bin/env python3
"""
L1 Instruction Generator using Qwen3.6-27B Model with vLLM
vLLM provides better compatibility with newer model architectures.
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/share/home/u19666033/dhj/DPed_pro/generate_l1_instructions_vllm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str = "/share/home/u19666033/dhj/models/Qwen3.6-27B"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class QwenInference:
    """Qwen model inference using vLLM."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.llm = None
        self.SamplingParams = None
        
    def load_model(self):
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
        except ImportError:
            logger.error("vLLM not installed. Please run: pip install vllm")
            sys.exit(1)
        
        logger.info(f"Loading Qwen model with vLLM from {self.config.model_path}...")
        
        self.llm = LLM(
            model=self.config.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=4096,
            dtype="bfloat16",
        )
        
        logger.info("Model loaded successfully with vLLM!")
    
    def generate_l1_instruction(self, original_instruction: str, scene_id: str = "") -> str:
        """Generate L1 instruction without pedestrian references."""
        system_prompt = """You are an expert in robot navigation instruction generation. Your task is to transform navigation instructions to Level 1 (L1) instructions.

L1 INSTRUCTIONS CRITERIA:
1. Focus ONLY on static environmental landmarks (hallway, kitchen, door, room, wall, window, staircase, furniture, etc.)
2. REMOVE ALL pedestrian-related content: avoid, person, pedestrian, people, man, woman, walking, standing, etc.
3. Keep the navigation goal and general route structure
4. Use natural, fluent English
5. Be concise but informative

Examples:
Input: "Turn left. No pedestrians. Move forward. Avoid the person near the wall. Turn right to the kitchen."
Output: "Turn left and proceed forward. Navigate to the kitchen area on the right."

Input: "Go past the staircase. Navigate carefully around the person walking ahead. Stop at the door."
Output: "Pass the staircase on your right. Continue to the doorway and stop."

CRITICAL: Output ONLY the rewritten instruction, nothing else."""

        user_prompt = f"""Transform this navigation instruction into an L1 instruction (no pedestrian references):

Original Instruction:
{original_instruction}

L1 Instruction (output only):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        sampling_params = self.SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        for tag in ['<think>', '</think>']:
            response = response.replace(tag, '')
        
        return response.strip()
    
    def generate_l1_instruction_strict(self, original_instruction: str, scene_id: str = "") -> str:
        """Generate L1 instruction with stricter prompt."""
        system_prompt = """You are a strict instruction rewriter. Remove ALL pedestrian content.

ABSOLUTE FORBIDDEN WORDS: person, people, pedestrian, avoid, wait, approaching, walking, standing, human, man, woman

Example:
Input: "Turn left. Avoid the person near the door. Walk forward to the kitchen."
Output: "Turn left and proceed forward. Navigate to the kitchen area."

Output ONLY the rewritten instruction."""

        user_prompt = f"Rewrite to remove ALL pedestrian content:\n{original_instruction}\n\nOutput:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        sampling_params = self.SamplingParams(
            temperature=0.3,
            top_p=0.8,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=1.1,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        for tag in ['<think>', '</think>']:
            response = response.replace(tag, '')
        
        return response.strip()


def contains_pedestrian_keywords(text: str) -> tuple:
    """Check if text contains pedestrian-related keywords."""
    pedestrian_keywords = [
        'person', 'persons', 'people', 'pedestrian', 'pedestrians',
        'man', 'woman', 'men', 'women',
        'human', 'humans', 'individual', 'individuals',
        'walking', 'walked', 'walks',
        'standing', 'stood', 'stands',
        'sitting', 'sat', 'sits',
        'approaching', 'approached',
        'passing', 'passed', 'passes',
        'moving', 'moved', 'moves',
        'waiting', 'waited', 'waits',
        'entering', 'entered', 'enters',
        'exiting', 'exited', 'exits',
        'crowd', 'crowded',
        'avoid', 'avoiding', 'evade', 'evading',
        'watch out for', 'look out for', 'be careful of',
        'give way to', 'make way for',
    ]
    
    ambiguous_words = ['head', 'ahead', 'behind']
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in pedestrian_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    person_refs = ['person', 'people', 'pedestrian', 'man', 'woman', 'human']
    for word in ambiguous_words:
        for person_ref in person_refs:
            if f"{word} toward {person_ref}" in text_lower:
                found_keywords.append(word)
            if f"{word} of {person_ref}" in text_lower:
                found_keywords.append(word)
            if f"{person_ref} {word}" in text_lower:
                found_keywords.append(word)
    
    return len(found_keywords) > 0, found_keywords


def validate_l1_instruction(l1_instruction: str, original_instruction: str = "") -> tuple:
    """Validate that the L1 instruction doesn't contain pedestrian-related content."""
    issues = []
    suggestions = []
    
    contains, found = contains_pedestrian_keywords(l1_instruction)
    if contains:
        issues.append(f"Contains pedestrian keywords: {found}")
        suggestions.append("Remove all references to people, pedestrians, avoidance actions")
    
    if len(l1_instruction.strip()) < 10:
        issues.append("Instruction too short or empty")
        suggestions.append("Provide a complete navigation instruction")
    
    if '?' in l1_instruction:
        issues.append("Contains question marks - should be imperative")
        suggestions.append("Rewrite as a command, not a question")
    
    nav_verbs = ['go', 'turn', 'move', 'walk', 'head', 'proceed', 
                  'continue', 'stop', 'wait', 'navigate', 'pass']
    has_nav_verb = any(verb in l1_instruction.lower() for verb in nav_verbs)
    if not has_nav_verb:
        issues.append("Missing navigation verbs")
        suggestions.append("Include directional commands like 'go', 'turn', 'move'")
    
    return len(issues) == 0, issues, suggestions[0] if suggestions else ""


def rewrite_instruction_with_fallback(model: QwenInference, original_instruction: str,
                                    scene_id: str = "", retry_count: int = 0,
                                    max_retries: int = 3) -> str:
    """Generate L1 instruction with retry logic and fallback."""
    if retry_count == 0:
        generated = model.generate_l1_instruction(original_instruction, scene_id)
    elif retry_count == 1:
        original_temp = model.config.temperature
        model.config.temperature = 0.5
        generated = model.generate_l1_instruction(original_instruction, scene_id)
        model.config.temperature = original_temp
    else:
        generated = model.generate_l1_instruction_strict(original_instruction, scene_id)
    
    is_valid, issues, suggestion = validate_l1_instruction(generated, original_instruction)
    
    if not is_valid and retry_count < max_retries - 1:
        logger.warning(f"Validation failed (attempt {retry_count + 1}): {issues}")
        return rewrite_instruction_with_fallback(model, original_instruction, scene_id, retry_count + 1, max_retries)
    
    if not is_valid:
        logger.warning(f"All retries failed. Using fallback.")
        return create_fallback_l1_instruction(original_instruction)
    
    return generated


def create_fallback_l1_instruction(original_instruction: str) -> str:
    """Create simplified L1 instruction by removing pedestrian-related content."""
    import re
    
    patterns_to_remove = [
        r',?\s*avoid\s+(the\s+)?(person|pedestrian|people|man|woman|human)[^\.]*',
        r',?\s*avoiding\s+(the\s+)?(person|pedestrian|people)[^\.]*',
        r',?\s*navigate\s+carefully\s+around\s+(the\s+)?(person|pedestrian|people)[^\.]*',
        r',?\s*wait\s+for\s+(the\s+)?(person|pedestrian|pedestrians)[^\.]*',
        r',?\s*stand\s+by[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+(walking|moving|standing|approaching|entering|exiting)[^\.]*',
        r',?\s*(walking|moving|standing|approaching)\s+(person|pedestrian|people)[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+on\s+(the\s+)?(left|right|ahead|behind)[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+near[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+in[^\.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+at[^\.]*',
        r'\.?\s*No\s+pedestrians\.?',
        r'\.?\s*No\s+people\.?',
        r'\.?\s*No\s+person\.?',
        r',?\s*give\s+way\s+to[^\.]*',
        r',?\s*make\s+way\s+for[^\.]*',
        r',?\s*watch\s+out\s+for[^\.]*',
        r',?\s*look\s+out\s+for[^\.]*',
    ]
    
    result = original_instruction
    for pattern in patterns_to_remove:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*', ', ', result)
    result = re.sub(r',\s*,', ',', result)
    result = result.strip()
    
    if result and not result.endswith('.'):
        result += '.'
    
    if len(result) < 10:
        result = "Proceed forward following the navigation route to the destination."
    
    return result


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
    """Save episodes to JSON file."""
    data = {"episodes": episodes}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_single_episode(episode: Dict[str, Any], model: QwenInference, 
                           dry_run: bool = False, max_retries: int = 3) -> Dict[str, Any]:
    """Process a single episode and generate L1 instruction."""
    original_instruction = episode.get('instruction', '')
    
    if not original_instruction:
        episode['instruction_l1'] = ''
        episode['l1_changed'] = False
        episode['l1_validation_passed'] = True
        return episode
    
    if not contains_pedestrian_keywords(original_instruction)[0]:
        episode['instruction_l1'] = original_instruction
        episode['l1_changed'] = False
        episode['l1_validation_passed'] = True
        return episode
    
    if dry_run:
        episode['instruction_l1'] = "[DRY RUN] " + original_instruction[:100] + "..."
        episode['l1_changed'] = True
        episode['l1_validation_passed'] = True
    else:
        try:
            l1_instruction = rewrite_instruction_with_fallback(
                model, original_instruction, 
                episode.get('scene_id', ''),
                max_retries=max_retries
            )
            
            is_valid, issues, _ = validate_l1_instruction(l1_instruction, original_instruction)
            
            episode['instruction_l1'] = l1_instruction
            episode['l1_changed'] = True
            episode['l1_validation_passed'] = is_valid
            
            if not is_valid:
                episode['l1_validation_issues'] = issues
                
        except Exception as e:
            logger.warning(f"Error generating L1: {e}")
            episode['instruction_l1'] = original_instruction
            episode['l1_changed'] = False
            episode['l1_error'] = str(e)
            episode['l1_validation_passed'] = True
    
    return episode


def process_file(input_filepath: str, output_filepath: str, model: QwenInference,
                 dry_run: bool = False, max_retries: int = 3, batch_size: int = 10) -> Dict[str, Any]:
    """Process a single data file and generate L1 instructions."""
    logger.info(f"Processing file: {input_filepath}")
    start_time = time.time()
    
    episodes = load_episodes_from_file(input_filepath)
    logger.info(f"Loaded {len(episodes)} episodes")
    
    if not episodes:
        return {"status": "empty", "episodes_processed": 0}
    
    results = []
    for i, episode in enumerate(episodes):
        result = process_single_episode(episode, model, dry_run, max_retries)
        results.append(result)
        if (i + 1) % batch_size == 0:
            logger.info(f"Processed {i + 1}/{len(episodes)} episodes")
    
    changed_count = sum(1 for ep in results if ep.get('l1_changed', False))
    save_episodes_to_file(output_filepath, results)
    
    elapsed = time.time() - start_time
    
    return {
        "status": "success",
        "input_file": input_filepath,
        "output_file": output_filepath,
        "episodes_processed": len(results),
        "instructions_changed": changed_count,
        "instructions_unchanged": len(results) - changed_count,
        "elapsed_seconds": elapsed
    }


def main():
    """Main function to process all training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate L1 instructions using Qwen model with vLLM")
    parser.add_argument("--input-dir", type=str, 
                       default="/share/home/u19666033/dhj/DPed_pro/dped_pro/train",
                       help="Input directory containing JSON files")
    parser.add_argument("--output-dir", type=str,
                       default="/share/home/u19666033/dhj/DPed_pro/dped_pro/train_l1",
                       help="Output directory for processed files")
    parser.add_argument("--model-path", type=str,
                       default="/share/home/u19666033/dhj/models/Qwen3.6-27B",
                       help="Path to Qwen model")
    parser.add_argument("--dry-run", action="store_true", help="Test without calling model")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = ModelConfig(model_path=args.model_path)
    model = QwenInference(config)
    
    if not args.dry_run:
        model.load_model()
    
    input_files = sorted([f for f in os.listdir(args.input_dir) 
                         if f.endswith('.json') or f.endswith('.json.gz')])
    
    if args.max_files > 0:
        input_files = input_files[:args.max_files]
    
    logger.info(f"Found {len(input_files)} files to process")
    
    all_stats = []
    processed_count = 0
    skipped_count = 0
    
    for filename in input_files:
        input_path = os.path.join(args.input_dir, filename)
        base_name = filename.replace('.gz', '')
        output_path = os.path.join(args.output_dir, base_name)
        
        if args.resume and os.path.exists(output_path):
            logger.info(f"Skipping already processed: {filename}")
            skipped_count += 1
            continue
        
        try:
            stats = process_file(input_path, output_path, model, args.dry_run, 
                               args.max_retries, args.batch_size)
            all_stats.append(stats)
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"Progress: {processed_count}/{len(input_files)} files processed")
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            all_stats.append({"input_file": filename, "status": "error", "error": str(e)})
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Files processed: {processed_count}, Skipped: {skipped_count}")
    
    if all_stats:
        total_episodes = sum(s.get('episodes_processed', 0) for s in all_stats)
        total_changed = sum(s.get('instructions_changed', 0) for s in all_stats)
        logger.info(f"Total episodes: {total_episodes}, Changed: {total_changed}")
    
    summary_path = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({"args": vars(args), "stats": all_stats}, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
