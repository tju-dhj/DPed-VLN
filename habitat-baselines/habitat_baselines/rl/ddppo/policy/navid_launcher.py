#!/usr/bin/env python3
"""Launcher for NaVid LoRA SFT training - passes sys.argv to train().

Imports NaVid model classes first to register with AutoModelForCausalLM,
then delegates to the shared LLaVA train() function.
"""
import sys, os
policy_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, policy_dir)

# Import navid model classes to register with transformers AutoModelForCausalLM
import navid.model  # noqa: F401

from unittest import mock
from llava.train.transformer_normalize_monkey_patch import patched_normalize
from accelerate.data_loader import BatchSamplerShard

def _batch_len(self):
    return len(self.batch_sampler)

def _batch_iter(self):
    return self.batch_sampler.__iter__()

with (
    mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
    mock.patch.object(BatchSamplerShard, "__len__", _batch_len),
    mock.patch.object(BatchSamplerShard, "__iter__", _batch_iter),
):
    from llava.train.train import train

if __name__ == "__main__":
    train()
