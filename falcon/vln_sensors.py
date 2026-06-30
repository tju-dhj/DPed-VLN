#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import attr
import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
from gym import spaces
if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.core.utils import not_none_validator

from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)


if TYPE_CHECKING:
    from omegaconf import DictConfig
@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None
@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None
@registry.register_sensor(name="FalconInstructionSensor")
class InstructionSensor(Sensor):
    """
    Sensor for natural language instructions in VLN tasks.
    Gets instruction from DynamicVLNCE dataset episodes.
    """
    cls_uuid: str = "falcon_instruction"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        # Use Box space for text instructions (encoded as integers)
        return spaces.Box(
            low=0,
            high=1000,  # vocabulary size
            shape=(512,),  # max instruction length
            dtype=int,
        )

    def get_observation(self, *args, episode=None, **kwargs):
        """Get instruction directly from episode."""
        # Return encoded instruction as tensor
        obs = torch.zeros(512, dtype=torch.long)
        
        if episode and hasattr(episode, 'instruction'):
            instruction = episode.instruction
            if isinstance(instruction, str):
                # Simple character-level encoding
                for i, char in enumerate(instruction[:512]):
                    obs[i] = ord(char) % 1000  # Map to vocabulary range
        
        return obs


@registry.register_sensor(name="FalconGTActionSensor")
class GT_ActionSensor(Sensor):
    """
    Sensor for ground truth actions in IL training.
    Gets GT actions from DynamicVLNCE dataset episodes.
    """
    cls_uuid: str = "falcon_gt_action"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=0,
            high=100,
            shape=(500,),  # Increased to accommodate longer action sequences
            dtype=int,
        )

    def get_observation(self, *args, episode=None, **kwargs):
        max_len = 500
        obs = torch.zeros(max_len, dtype=torch.long)
        if episode and hasattr(episode, 'gt_action'):
            gt_actions = episode.gt_action
            n = min(len(gt_actions), max_len)
            if n > 0:
                obs[:n] = torch.tensor(gt_actions[:n], dtype=torch.long)
        return obs

@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "goal_format", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor(name="FalconStartingPointGpsCompassSensor")
class StartingPointGPSCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "starting_point_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.start_position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )

# @registry.register_sensor(name="FalconStartingPointGpsCompassSensor")
# class StartingPointGPSCompassSensor(Sensor):
#     """
#     Sensor that provides robot's position and heading relative to the starting point.

#     This sensor replaces the original pointgoal_with_gps_compass sensor which provides
#     the goal position. This new sensor provides the robot's current position relative
#     to its starting point, allowing the network to track its displacement from the start
#     without knowing the absolute goal position.

#     Returns: [distance_from_start, heading_relative_to_start]
#         - distance_from_start: Euclidean distance from starting point (meters)
#         - heading_relative_to_start: Angle from starting point to current position (radians)
#     """
#     cls_uuid: str = "starting_point_gps_compass"

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._sim = None
#         self._start_position = None
#         self._start_heading = None

#     def set_sim(self, sim):
#         """Set the simulator reference after initialization."""
#         self._sim = sim

#     def _get_uuid(self, *args, **kwargs):
#         return self.cls_uuid

#     def _get_sensor_type(self, *args, **kwargs):
#         return SensorTypes.SEMANTIC

#     def _get_observation_space(self, *args, **kwargs):
#         return spaces.Box(
#             low=0,
#             high=1e6,
#             shape=(2,),
#             dtype=np.float32,
#         )

#     def get_observation(self, *args, episode=None, **kwargs):
#         """
#         Returns robot's current position relative to starting point.
#         """
#         if self._sim is None:
#             return np.array([0.0, 0.0], dtype=np.float32)

#         try:
#             agent_state = self._sim.get_agent_state(0)
#             ###todotodo
#             if agent_state is not None:
#                 current_position = agent_state.position
#                 current_heading = agent_state.rotation.euler_angles[2] if hasattr(agent_state.rotation, 'euler_angles') else 0.0

#                 if self._start_position is None:
#                     self._start_position = np.array([
#                         current_position[0],
#                         current_position[2]
#                     ])
#                     self._start_heading = current_heading

#                 current_xy = np.array([
#                     current_position[0],
#                     current_position[2]
#                 ])
#                 distance = np.linalg.norm(current_xy - self._start_position)

#                 delta = current_xy - self._start_position
#                 angle = np.arctan2(delta[1], delta[0])

#                 relative_heading = angle - self._start_heading

#                 return np.array([distance, relative_heading], dtype=np.float32)
#         except Exception:
#             pass

#         return np.array([0.0, 0.0], dtype=np.float32)

#     def reset(self):
#         """Reset starting position for new episode."""
#         self._start_position = None
#         self._start_heading = None


@registry.register_sensor(name="DynamicVLNCEEpisodeSensor")
class DynamicVLNCEEpisodeSensor(Sensor):
    """
    Sensor for DynamicVLNCE episode metadata.
    Stores episode information and coordinates with other VLN sensors.
    """
    cls_uuid: str = "dynamic_vlnce_episode"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instruction_sensor = None
        self._gt_action_sensor = None
        self._starting_point_sensor = None
        self._current_episode = None

    def set_sensor_references(self, instruction_sensor, gt_action_sensor, starting_point_sensor=None):
        """Set references to other VLN sensors for coordination."""
        self._instruction_sensor = instruction_sensor
        self._gt_action_sensor = gt_action_sensor
        self._starting_point_sensor = starting_point_sensor

    def set_episode(self, episode):
        """Set the current episode."""
        self._current_episode = episode

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(self, *args, episode=None, **kwargs):
        """Return episode metadata."""
        return np.array([1.0], dtype=np.float32)