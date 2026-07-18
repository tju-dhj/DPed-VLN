#!/usr/bin/env python3
"""Launcher for NaviLLa LoRA training that fixes relative imports for torchrun."""
import sys
import os

# Set up the package context so relative imports work
NAVILA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVILA_ROOT = os.path.join(NAVILA_ROOT, "habitat-baselines/habitat_baselines/rl/ddppo/policy/navila")
sys.path.insert(0, NAVILA_ROOT)
sys.path.insert(0, os.path.dirname(NAVILA_ROOT))  # parent too

# Now import and run train_mem
from llava.train.train_mem import train

if __name__ == "__main__":
    from unittest import mock
    from llava.train.transformer_normalize_monkey_patch import patched_normalize
    
    def __len__(self):
        return len(self.batch_sampler)
    
    def __iter__(self):
        return self.batch_sampler.__iter__()
    
    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=__len__),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=__iter__),
    ):
        train()
