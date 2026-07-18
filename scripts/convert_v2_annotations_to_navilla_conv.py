#!/usr/bin/env python3
"""Convert annotations.json to navilla_conversations.json for LLaVA SFT training."""
import argparse
import json

ACTION_ID_TO_TEXT = {0: "STOP", 1: "MOVE FORWARD", 2: "TURN LEFT", 3: "TURN RIGHT"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()

    with open(args.src) as f:
        data = json.load(f)

    samples = []
    for item in data:
        actions = item.get("actions", [])
        if not actions:
            continue
        instruction = item["instructions"][0] if item.get("instructions") else "Navigate to the target location."
        action_texts = [ACTION_ID_TO_TEXT.get(a, "STOP") for a in actions]
        sample = {
            "id": item["id"],
            "video": item["video"],
            "conversations": [
                {"from": "human", "value": f"<video>\n{instruction}"},
                {"from": "gpt", "value": ", ".join(action_texts)},
            ],
        }
        samples.append(sample)

    with open(args.dst, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(samples)} samples → {args.dst}")

if __name__ == "__main__":
    main()
