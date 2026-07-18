#!/usr/bin/env python3
"""
Prepare sampled annotations for StreamVLN/NaVILA training from DPed_VLN sampled_data.
Samples 3000 episodes from the full dataset for faster training.
"""

import os
import json
import gzip
import random
from pathlib import Path

def load_sampled_episodes(data_root, level="v1"):
    """Load all episodes from sampled .json.gz files."""
    sampled_dir = Path(data_root) / "sampled_data" / level / "train"
    all_episodes = []
    
    for json_gz in sorted(sampled_dir.glob("*.json.gz")):
        with gzip.open(json_gz, 'rt') as f:
            data = json.load(f)
            episodes = data.get("episodes", [])
            for ep in episodes:
                ep["scene_id"] = json_gz.stem  # scene name without .json.gz
            all_episodes.extend(episodes)
    
    return all_episodes

def create_streamvln_annotations(episodes, output_path, video_base="v1/train"):
    """Create annotations.json for StreamVLN training."""
    annotations = []
    for idx, ep in enumerate(episodes):
        scene = ep.get("scene_id", "")
        episode_id = ep.get("episode_id", idx)
        # Video path format: v1/train/{scene}_{episode_id}
        video = f"{video_base}/{scene}_{episode_id}"
        
        instructions = []
        if "instruction" in ep:
            instructions.append(ep["instruction"])
        elif "instructions" in ep:
            instructions = ep["instructions"]
        
        actions = ep.get("action_sequence", [])
        
        annotations.append({
            "id": idx,
            "video": video,
            "instructions": instructions,
            "actions": actions
        })
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f)
    print(f"Created {output_path} with {len(annotations)} entries")

def create_navilla_conversations(episodes, output_path, video_base="v1/train"):
    """Create navilla_conversations.json for NaVILA training."""
    action_map = {0: "STOP", 1: "MOVE FORWARD", 2: "TURN LEFT", 3: "TURN RIGHT"}
    
    conversations = []
    for idx, ep in enumerate(episodes):
        scene = ep.get("scene_id", "")
        episode_id = ep.get("episode_id", idx)
        video = f"{video_base}/{scene}_{episode_id}"
        
        instructions = []
        if "instruction" in ep:
            instructions.append(ep["instruction"])
        elif "instructions" in ep:
            instructions = ep["instructions"]
        
        instruction_text = instructions[0] if instructions else "Navigate to the goal."
        
        # Convert action indices to text
        actions = ep.get("action_sequence", [])
        action_texts = [action_map.get(a, str(a)) for a in actions]
        response_text = ", ".join(action_texts)
        
        conversation = {
            "id": idx,
            "video": video,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{instruction_text}"
                },
                {
                    "from": "gpt", 
                    "value": response_text
                }
            ]
        }
        conversations.append(conversation)
    
    with open(output_path, 'w') as f:
        json.dump(conversations, f)
    print(f"Created {output_path} with {len(conversations)} entries")

def create_navid_conversations(episodes, output_path, video_base="v1/train"):
    """Create navid_conversations.json for NaVid training."""
    action_map = {0: "STOP", 1: "MOVE FORWARD", 2: "TURN LEFT", 3: "TURN RIGHT"}
    
    conversations = []
    for idx, ep in enumerate(episodes):
        scene = ep.get("scene_id", "")
        episode_id = ep.get("episode_id", idx)
        video = f"{video_base}/{scene}_{episode_id}"
        
        instructions = []
        if "instruction" in ep:
            instructions.append(ep["instruction"])
        elif "instructions" in ep:
            instructions = ep["instructions"]
        
        instruction_text = instructions[0] if instructions else "Navigate to the goal."
        
        actions = ep.get("action_sequence", [])
        action_texts = [action_map.get(a, str(a)) for a in actions]
        response_text = ", ".join(action_texts)
        
        conversation = {
            "id": idx,
            "video": video,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{instruction_text}"
                },
                {
                    "from": "gpt",
                    "value": response_text
                }
            ]
        }
        conversations.append(conversation)
    
    with open(output_path, 'w') as f:
        json.dump(conversations, f)
    print(f"Created {output_path} with {len(conversations)} entries")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/share/home/u19666033/dhj/dped-vln/DPed_VLN")
    parser.add_argument("--level", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--max_episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    output_dir = Path(args.data_root) / f"streamvln_training_data_{args.level}_sampled_3000"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video folder structure (symlink to original)
    video_src = Path(args.data_root) / f"streamvln_training_data_{args.level}" / args.level
    video_dst = output_dir / args.level
    if not video_dst.exists():
        video_dst.symlink_to(video_src, target_is_directory=True)
        print(f"Created symlink: {video_dst} -> {video_src}")
    
    # Load episodes
    print(f"Loading episodes from {args.data_root}/sampled_data/{args.level}/train/...")
    episodes = load_sampled_episodes(args.data_root, args.level)
    print(f"Total episodes loaded: {len(episodes)}")
    
    # Sample
    if len(episodes) > args.max_episodes:
        episodes = random.sample(episodes, args.max_episodes)
        print(f"Sampled {len(episodes)} episodes")
    
    # Create annotations
    annotations_path = output_dir / "annotations.json"
    create_streamvln_annotations(episodes, annotations_path, video_base=f"{args.level}/train")
    
    navilla_path = output_dir / "navilla_conversations.json"
    create_navilla_conversations(episodes, navilla_path, video_base=f"{args.level}/train")
    
    print(f"\nDone! Data prepared at: {output_dir}")
    print(f"  - annotations.json: {annotations_path}")
    print(f"  - navilla_conversations.json: {navilla_path}")
    print(f"  - video folder: {video_dst}")

if __name__ == "__main__":
    main()
