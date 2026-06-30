#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from gym import spaces

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorSuite
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.rearrange_sim import (
    RearrangeSim,
    add_perf_timing_func,
)
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    CollisionDetails,
    UsesArticulatedAgentInterface,
    rearrange_collision,
    rearrange_logger,
)
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
import magnum as mn

import sys
import os
# Add falcon path to import vln_sensors
# dynamic_vlnce_task.py is in: habitat-lab/habitat/tasks/rearrange/dynamic_vlnce/
# Need to go up 6 levels to reach dped_pro, then add 'falcon'
_dped_pro_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
_falcon_path = os.path.join(_dped_pro_path, 'falcon')
if _falcon_path not in sys.path:
    sys.path.insert(0, _falcon_path)

from vln_sensors import (
    InstructionSensor,
    GT_ActionSensor,
    DynamicVLNCEEpisodeSensor,
    StartingPointGPSCompassSensor,
)


def quaternion_to_rad_angle(source_rotation):
    rad_angle = 2 * np.arctan2(np.sqrt(source_rotation[1]**2 + source_rotation[2]**2 + source_rotation[3]**2), source_rotation[0])
    return rad_angle


@registry.register_task(name="DynamicVLNCETask-v0")
class DynamicVLNCETask(NavigationTask):
    """
    Dynamic VLN-CE Task that integrates instruction and GT action data
    from DynamicVLNCE dataset episodes.
    """

    _cur_episode_step: int
    _articulated_agent_pos_start: Dict[str, Tuple[np.ndarray, float]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize VLN sensors
        self._instruction_sensor = None
        self._gt_action_sensor = None
        self._episode_sensor = None
        self._starting_point_gps_compass_sensor = None

        # Current episode data
        self._current_episode = None
        self._current_step = 0

        # Add VLN sensors to the task's sensor suite
        # Note: self.sensor_suite is set by the parent class __init__
        if hasattr(self, 'sensor_suite') and self.sensor_suite is not None:
            self._init_vln_sensors(self.sensor_suite)

    def _init_vln_sensors(self, sensor_suite: SensorSuite) -> None:
        """
        Initialize VLN sensors and add them to the sensor suite.
        This is called during __init__ to ensure sensors are available for all configurations.
        """
        # Only add sensors that don't already exist in the suite
        # This prevents overwriting sensors that were already added via YAML config

        # Add instruction sensor if not present
        if "instruction" not in sensor_suite.sensors:
            sensor = InstructionSensor()
            sensor_suite.sensors["instruction"] = sensor
            sensor_suite.observation_spaces.spaces["instruction"] = sensor.observation_space

        # Add GT action sensor if not present
        if "gt_action" not in sensor_suite.sensors:
            sensor = GT_ActionSensor()
            sensor_suite.sensors["gt_action"] = sensor
            sensor_suite.observation_spaces.spaces["gt_action"] = sensor.observation_space

        # Add episode sensor if not present
        if "dynamic_vlnce_episode" not in sensor_suite.sensors:
            sensor = DynamicVLNCEEpisodeSensor()
            sensor_suite.sensors["dynamic_vlnce_episode"] = sensor
            sensor_suite.observation_spaces.spaces["dynamic_vlnce_episode"] = sensor.observation_space

        # Add starting point GPS compass sensor if not present
        if "agent_0_starting_point_gps_compass" not in sensor_suite.sensors:
            sensor = StartingPointGPSCompassSensor()
            sensor_suite.sensors["agent_0_starting_point_gps_compass"] = sensor
            sensor_suite.observation_spaces.spaces["agent_0_starting_point_gps_compass"] = sensor.observation_space

        # Store references
        self._instruction_sensor = sensor_suite.sensors["instruction"]
        self._gt_action_sensor = sensor_suite.sensors["gt_action"]
        self._episode_sensor = sensor_suite.sensors["dynamic_vlnce_episode"]
        self._starting_point_gps_compass_sensor = sensor_suite.sensors["agent_0_starting_point_gps_compass"]

        # Set sim reference for StartingPointGPSCompassSensor
        if hasattr(self, '_sim') and self._sim is not None:
            if self._starting_point_gps_compass_sensor._sim is None:
                self._starting_point_gps_compass_sensor._sim = self._sim

        # Set up sensor coordination
        if self._episode_sensor:
            self._episode_sensor.set_sensor_references(
                self._instruction_sensor,
                self._gt_action_sensor,
                self._starting_point_gps_compass_sensor
            )

    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Modifies the sensor suite in place to duplicate articulated agent specific sensors.
        Only adds sensors that don't already exist.
        """
        # Add instruction sensor if not present
        if "instruction" not in sensor_suite.sensors:
            sensor = InstructionSensor()
            sensor_suite.sensors["instruction"] = sensor
            sensor_suite.observation_spaces.spaces["instruction"] = sensor.observation_space

        # Add GT action sensor if not present
        if "gt_action" not in sensor_suite.sensors:
            sensor = GT_ActionSensor()
            sensor_suite.sensors["gt_action"] = sensor
            sensor_suite.observation_spaces.spaces["gt_action"] = sensor.observation_space

        # Add episode sensor if not present
        if "dynamic_vlnce_episode" not in sensor_suite.sensors:
            sensor = DynamicVLNCEEpisodeSensor()
            sensor_suite.sensors["dynamic_vlnce_episode"] = sensor
            sensor_suite.observation_spaces.spaces["dynamic_vlnce_episode"] = sensor.observation_space

        # Add starting point GPS compass sensor if not present
        if "agent_0_starting_point_gps_compass" not in sensor_suite.sensors:
            sensor = StartingPointGPSCompassSensor()
            sensor_suite.sensors["agent_0_starting_point_gps_compass"] = sensor
            sensor_suite.observation_spaces.spaces["agent_0_starting_point_gps_compass"] = sensor.observation_space

        # Store references for coordination
        self._instruction_sensor = sensor_suite.sensors["instruction"]
        self._gt_action_sensor = sensor_suite.sensors["gt_action"]
        self._episode_sensor = sensor_suite.sensors["dynamic_vlnce_episode"]
        self._starting_point_gps_compass_sensor = sensor_suite.sensors["agent_0_starting_point_gps_compass"]

        # Set sim reference for StartingPointGPSCompassSensor
        if self._starting_point_gps_compass_sensor and hasattr(self, '_sim') and self._sim is not None:
            if self._starting_point_gps_compass_sensor._sim is None:
                self._starting_point_gps_compass_sensor._sim = self._sim

        # Set up sensor coordination
        self._episode_sensor.set_sensor_references(
            self._instruction_sensor,
            self._gt_action_sensor,
            self._starting_point_gps_compass_sensor
        )

    def reset(self, episode: Episode):
        """
        Reset the task with a new episode.
        """
        observations = super().reset(episode)

        # Store current episode
        self._current_episode = episode
        self._current_step = 0

        # Update sim reference in case it changed (important for vector env)
        if self._starting_point_gps_compass_sensor and hasattr(self, '_sim') and self._sim is not None:
            if self._starting_point_gps_compass_sensor._sim is None:
                self._starting_point_gps_compass_sensor._sim = self._sim
            self._starting_point_gps_compass_sensor.reset()

        # Update episode sensor with current episode data
        if self._episode_sensor:
            self._episode_sensor.set_episode(episode)

        # Reset GT action sensor
        if self._gt_action_sensor:
            self._gt_action_sensor.reset_action_index()

        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        """
        Step the task with the given action.
        """
        observations = super().step(action, episode)
        
        # Advance GT action sensor if available
        if self._gt_action_sensor:
            self._gt_action_sensor.advance_action()
        
        self._current_step += 1
        
        return observations

    def get_current_instruction(self) -> str:
        """Get the current instruction text."""
        if self._current_episode and hasattr(self._current_episode, 'instruction'):
            return self._current_episode.instruction
        return ""

    def get_current_gt_action(self) -> int:
        """Get the current GT action to execute."""
        if self._gt_action_sensor:
            return self._gt_action_sensor.get_current_action()
        return None

    def has_more_gt_actions(self) -> bool:
        """Check if there are more GT actions in the sequence."""
        if self._gt_action_sensor:
            return self._gt_action_sensor.has_more_actions()
        return False

    def get_episode_info(self) -> Dict[str, Any]:
        """Get information about the current episode."""
        if not self._current_episode:
            return {}
        
        # 优先使用 original_episode_id（包含_v1/_v2后缀），如果没有则使用episode_id
        episode_id = getattr(self._current_episode, 'original_episode_id', None)
        if not episode_id:
            episode_id = getattr(self._current_episode, 'episode_id', '')
        
        info = {
            "episode_id": episode_id,
            "original_episode_id": getattr(self._current_episode, 'original_episode_id', episode_id),
            "scene_id": getattr(self._current_episode, 'scene_id', ''),
            "instruction": getattr(self._current_episode, 'instruction', ''),
            "instruction_tokens": getattr(self._current_episode, 'instruction_tokens', []),
            "gt_action_length": len(getattr(self._current_episode, 'gt_action', [])),
            "current_step": self._current_step,
            "instruction_source": getattr(self._current_episode, 'instruction_source', ''),
            "has_gt_actions": len(getattr(self._current_episode, 'gt_action', [])) > 0,
        }
        
        return info

    def is_episode_active(self) -> bool:
        """Check if the current episode is still active."""
        return self._current_episode is not None

    def get_instruction_tokens(self) -> List[str]:
        """Get the current instruction tokens."""
        if self._current_episode and hasattr(self._current_episode, 'instruction_tokens'):
            return self._current_episode.instruction_tokens
        return []

    def get_gt_action_sequence(self) -> List[int]:
        """Get the complete GT action sequence."""
        if self._current_episode and hasattr(self._current_episode, 'gt_action'):
            return self._current_episode.gt_action
        return []

    def get_remaining_gt_actions(self) -> List[int]:
        """Get the remaining GT actions in the sequence."""
        if self._gt_action_sensor and self._gt_action_sensor._current_gt_actions:
            current_idx = self._gt_action_sensor._current_action_index
            return self._gt_action_sensor._current_gt_actions[current_idx:]
        return []

    def set_instruction(self, instruction: str, tokens: List[str] = None):
        """Manually set the instruction for the current episode."""
        if self._instruction_sensor:
            self._instruction_sensor.set_instruction(instruction, tokens)

    def set_gt_actions(self, gt_actions: List[int]):
        """Manually set the GT actions for the current episode."""
        if self._gt_action_sensor:
            self._gt_action_sensor.set_gt_actions(gt_actions)

    def reset_gt_action_index(self):
        """Reset the GT action index to the beginning."""
        if self._gt_action_sensor:
            self._gt_action_sensor.reset_action_index()

    def get_sensor_observations(self) -> Dict[str, Any]:
        """Get observations from all VLN sensors."""
        observations = {}
        
        if self._instruction_sensor:
            observations["instruction"] = self._instruction_sensor.get_observation({})
        
        if self._gt_action_sensor:
            observations["gt_action"] = self._gt_action_sensor.get_observation({})
        
        if self._episode_sensor:
            observations["dynamic_vlnce_episode"] = self._episode_sensor.get_observation({})
        
        return observations

    def _get_episode_over_success(self, episode: Episode) -> float:
        """
        Returns 1.0 if the agent has reached the goal, 0.0 otherwise.
        """
        return float(self._is_episode_over_success())

    def _is_episode_over_success(self) -> bool:
        """
        Check if the episode is over due to success.
        """
        # Use the parent class success check
        return super()._is_episode_over_success()

    def _get_episode_over_timeout(self, episode: Episode) -> float:
        """
        Returns 1.0 if the episode is over due to timeout, 0.0 otherwise.
        """
        return float(self._is_episode_over_timeout())

    def _is_episode_over_timeout(self) -> bool:
        """
        Check if the episode is over due to timeout.
        """
        # Use the parent class timeout check
        return super()._is_episode_over_timeout()

    def _get_episode_over_failure(self, episode: Episode) -> float:
        """
        Returns 1.0 if the episode is over due to failure, 0.0 otherwise.
        """
        return float(self._is_episode_over_failure())

    def _is_episode_over_failure(self) -> bool:
        """
        Check if the episode is over due to failure.
        """
        # Use the parent class failure check
        return super()._is_episode_over_failure()

    def _get_episode_over(self, episode: Episode) -> float:
        """
        Returns 1.0 if the episode is over, 0.0 otherwise.
        """
        return float(self._is_episode_over())

    def _is_episode_over(self) -> bool:
        """
        Check if the episode is over.
        """
        # Use the parent class episode over check
        return super()._is_episode_over()

    def get_reward(self, observations, action, is_episode_done):
        """
        Get the reward for the current step.
        """
        # Use the parent class reward calculation
        return super().get_reward(observations, action, is_episode_done)

    def get_done(self, observations):
        """
        Get the done status for the current step.
        """
        # Use the parent class done calculation
        return super().get_done(observations)

    def get_info(self, observations, action):
        """
        Get additional information for the current step.
        """
        info = super().get_info(observations, action)
        
        # Add VLN-specific information
        vln_info = self.get_episode_info()
        info.update(vln_info)
        
        return info