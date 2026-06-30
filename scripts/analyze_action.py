from collections import Counter
import json
import glob
import gzip
import os

data_root = "/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit/train"

counter = Counter()
episode_count = 0
file_count = 0
total = 0
bad_files = 0
missing_action_eps = 0

paths = sorted(glob.glob(os.path.join(data_root, "*.json")) +
               glob.glob(os.path.join(data_root, "*.json.gz")))

print("num files:", len(paths))

for path in paths:
    try:
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        file_count += 1

        # 兼容两种格式：
        # 1) {"episodes": [ep1, ep2, ...]}
        # 2) 单个 episode dict
        if isinstance(data, dict) and "episodes" in data:
            episodes = data["episodes"]
        elif isinstance(data, list):
            episodes = data
        else:
            episodes = [data]

        for ep in episodes:
            episode_count += 1

            acts = (
                ep.get("gt_action")
                or ep.get("action")
                or ep.get("oracle_actions")
            )

            if acts is None:
                missing_action_eps += 1
                continue

            # 和你训练代码一致，最多截断 400 步
            for a in acts[:400]:
                a = int(a)
                if 0 <= a < 4:
                    counter[a] += 1
                    total += 1
                else:
                    # 如果你怀疑有 4/5 动作，也可以单独统计
                    counter[f"invalid_{a}"] += 1

    except Exception as e:
        bad_files += 1
        print("[BAD FILE]", path, e)

print("=" * 60)
print("file_count:", file_count)
print("episode_count:", episode_count)
print("missing_action_eps:", missing_action_eps)
print("bad_files:", bad_files)
print("total_valid_actions:", total)
print("counter:", counter)

if total > 0:
    ratio = {k: v / total for k, v in counter.items() if isinstance(k, int)}
    print("ratio:", ratio)
    print("majority baseline:", max(v for k, v in counter.items() if isinstance(k, int)) / total)