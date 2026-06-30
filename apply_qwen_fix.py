#!/usr/bin/env python3
"""Apply all qwen3_5 patches for transformers 4.57.1 compatibility."""
import os, re

SITE = "/share/home/u19666033/.conda/envs/vllm/lib/python3.9/site-packages"

def fix_file(path):
    with open(path) as f:
        content = f.read()

    original = content

    # 1. Add future annotations import after license header
    if "from __future__ import annotations" not in content:
        # Find the first import after license comments
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#') or line.strip() == '':
                continue
            if line.startswith('from ') or line.startswith('import '):
                insert_idx = i
                break
        lines.insert(insert_idx, 'from __future__ import annotations')
        content = '\n'.join(lines)

    # 2. Fix PreTrainedConfig -> PretrainedConfig
    content = content.replace('PreTrainedConfig', 'PretrainedConfig')

    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print(f"Fixed: {path}")
    else:
        print(f"No changes: {path}")


if __name__ == "__main__":
    for model in ['qwen3_5', 'qwen3_5_moe']:
        cfg = os.path.join(SITE, f'transformers/models/{model}/configuration_{model}.py')
        if os.path.exists(cfg):
            fix_file(cfg)
        else:
            print(f"File not found: {cfg}")

    # Also patch modeling files that import from config
    for model in ['qwen3_5', 'qwen3_5_moe']:
        mod = os.path.join(SITE, f'transformers/models/{model}/modeling_{model}.py')
        if os.path.exists(mod):
            with open(mod) as f:
                content = f.read()
            if 'PreTrainedConfig' in content:
                content = content.replace('PreTrainedConfig', 'PretrainedConfig')
                with open(mod, 'w') as f:
                    f.write(content)
                print(f"Fixed modeling: {mod}")
            else:
                print(f"No changes: {mod}")
