#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVLN vendored package wrapper.

注意：这里不要在 import package 时就强制导入子模块（例如 action_parser），
否则在某些 sys.path/多份代码共存场景下容易触发循环导入或导入到错误路径。
需要时请从 `habitat_baselines.rl.ddppo.policy.streamvln.action_parser` 显式导入。
"""

__all__ = []

