#!/usr/bin/env python3
"""Convert DPed v2 annotations.json → navilla_conversations.json format."""
import json, os

ACTION_NAMES = {0: "STOP", 1: "MOVE FORWARD", 2: "TURN LEFT", 3: "TURN RIGHT"}

def main():
    level = os.environ.get("LEVEL", "v2")
    base = f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_{level}"
    src = os.path.join(base, "annotations.json")
    dst = os.path.join(base, "navilla_conversations.json")

    with open(src) as f:
        data = json.load(f)

    output = []
    for item in data:
        actions = item.get("actions", [])
        if len(actions) < 1:
            continue
        instruction = item.get("instructions", [""])
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        action_str = ", ".join(ACTION_NAMES.get(a, "STOP") for a in actions)
        output.append({
            "id": item["id"],
            "video": item["video"],
            "conversations": [
                {"from": "human", "value": f"<video>\n{instruction}"},
                {"from": "gpt", "value": action_str},
            ],
        })

    with open(dst, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Converted {len(output)} samples → {dst}")

if __name__ == "__main__":
    main()
