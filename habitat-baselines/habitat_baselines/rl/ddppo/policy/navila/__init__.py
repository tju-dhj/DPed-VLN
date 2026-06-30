#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Integration for Falcon Framework (Habitat3)

This package integrates the NaVILA (Navigation with Vision-Language Actions) model
into the Falcon framework for social navigation tasks in Habitat3.

Modules:
- action_parser: Parses language instructions into discrete actions
- navila_policy: NaVILA policy implementation (currently not used, see navila_evaluator)
- llava: LLAVA vision-language model

Note: The actual NaVILA inference is handled by the NaVILAEvaluator in
habitat_baselines.rl.ppo.navila_evaluator, as the language-based action generation
doesn't fit naturally into the standard policy network paradigm.
"""

from .action_parser import NaVILAActionParser

__all__ = ["NaVILAActionParser"]