from enum import Enum
import os
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from gymnasium import spaces
from pyrobosim_msgs.msg import TaskAction, WorldState

from .pyrobosim_ros_env import PyRoboSimRosEnv
from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import TaskAction, WorldState
from pyrobosim_msgs.srv import RequestWorldState, ResetWorld


def _dist(a: Point, b: Point) -> float:
    """Calculate distance between two (geometry_msgs.msg) Points."""
    return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))


class GreenhouseEnv(PyRoboSimRosEnv):
    sub_types = Enum("sub_types", "Plain Battery Random")

    @classmethod
    def get_world_file_path(cls, sub_type: sub_types) -> str:
        """Get the world file path for a given subtype."""
        if sub_type == GreenhouseEnv.sub_types.Plain:
            return os.path.join("rl_ws_worlds", "worlds", "greenhouse_plain.yaml")
        elif sub_type == GreenhouseEnv.sub_types.Battery:
            return os.path.join("rl_ws_worlds", "worlds", "greenhouse_battery.yaml")
        elif sub_type == GreenhouseEnv.sub_types.Random:
            return os.path.join("rl_ws_worlds", "worlds", "greenhouse_random.yaml")
        else:
            raise ValueError(f"Invalid environment: {sub_type}")

    def __init__(
        self,
        sub_type: sub_types,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
    ):
        """
        Instantiate Greenhouse environment.

        :param sub_type: Subtype of this environment, e.g. `GreenhouseEnv.sub_types.Deterministic`.
        :param node: Node instance needed for ROS communication.
        :param max_steps_per_episode: Limit the steps (when to end the episode).
        :param realtime: Whether actions take time.
        :param discrete_actions: Choose discrete actions (needed for DQN).
        """
        if sub_type == GreenhouseEnv.sub_types.Plain:
            # All plants are in their places
            pass
        elif sub_type == GreenhouseEnv.sub_types.Random:
            # Plants are randomly across tables
            pass
        elif sub_type == GreenhouseEnv.sub_types.Battery:
            # Battery (= water) is limited
            pass
        else:
            raise ValueError(f"Invalid environment: {sub_type}")
        self.sub_type = sub_type

        super().__init__(
            node,
            None,  # reward_fn
            None,  # reset_validation_fn
            max_steps_per_episode,
            realtime,
            discrete_actions,
        )

        # Observation space is defined by:
        self.max_n_objects = 3
        self.max_dist = 10
        # array of n objects with a class and distance each, plus battery level
        self.obs_size = 2 * self.max_n_objects + 2
        low = -1.0 * np.ones(self.obs_size, dtype=np.float32)
        low[-2:] = 0.0
        high = np.ones(self.obs_size, dtype=np.float32)  # max class = 1
        high[1::2] = self.max_dist
        high[-2:] = 1.0
        self.observation_space = spaces.Box(low=low, high=high)
        print(f"{self.observation_space=}")

        self.plants = [obj.name for obj in self.world_state.objects]
        # print(f"{self.plants=}")
        self.good_plants = [
            obj.name for obj in self.world_state.objects if obj.category == "plant_good"
        ]
        # print(f"{self.good_plants=}")

        self.waypoints = [
            "table_c",
            "table_ne",
            "table_e",
            "table_se",
            "table_s",
            "table_sw",
            "table_w",
            "table_nw",
            "table_n",
        ]

    def _action_space(self):
        if self.sub_type in (
            GreenhouseEnv.sub_types.Plain,
            GreenhouseEnv.sub_types.Random,
        ):
            self.num_actions = 2  # stay ducked or water plant
        else:
            self.num_actions = 3  # stay ducked, water plant, or go charge

        if self.discrete_actions:
            return spaces.Discrete(self.num_actions)
        else:
            return spaces.Box(
                low=np.zeros(self.num_actions, dtype=np.float32),
                high=np.ones(self.num_actions, dtype=np.float32),
            )

    def step(self, action):
        info = {}
        truncated = self.step_number >= self.max_steps_per_episode
        if truncated:
            print(
                f"Maximum steps ({self.max_steps_per_episode}) exceeded. "
                f"Truncated episode with watered fraction {self.watered_plant_fraction()}."
            )

        # print(f"{'*'*10}")
        # print(f"{action=}")
        if self.discrete_actions:
            action = float(action)
        else:
            action = np.argmax(action)

        if action == 1:  # water a plant
            self.mark_table(self.get_current_location())
        elif action == 2:  # charge
            self.go_to_charger()

        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        self.world_state = future.result().state

        self.step_number += 1
        reward, terminated = self._calculate_reward(action)
        # print(f"{reward=}")

        if not terminated:
            self.go_to_next_wp()
            # action_result = result_future.result().result

        observation = self._get_obs()  # update self.world_state
        # print(f"{observation=}")

        return observation, reward, terminated, truncated, info

    def mark_table(self, loc):
        close_goal = ExecuteTaskAction.Goal()
        close_goal.action = TaskAction()
        close_goal.action.robot = "robot"
        close_goal.action.type = "close"
        close_goal.action.target_location = loc

        goal_future = self.execute_action_client.send_goal_async(close_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

    def _calculate_reward(self, action):
        # Calculate reward
        reward = 0.0
        terminated = False
        plants_by_distance = self._get_plants_by_distance(self.world_state)
        # print(f"{action=}")

        close_radius = 1.0
        self.is_dead = False

        if self.battery_level() <= 0.0:
            print(
                "🪫 Ran out of battery. "
                f"Terminated in {self.step_number} steps "
                f"with watered fraction {self.watered_plant_fraction()}."
            )
            self.is_dead = True
            return -5.0, True

        if action == 0:  # stay ducked
            # Robot gets a penalty if it decides to ignore a waterable plant.
            robot_location = self.world_state.robots[0].last_visited_location
            for plant in self.world_state.objects:
                if (plant.category == "plant_good") and (
                    plant.parent == robot_location
                ):
                    for location in self.world_state.locations:
                        if robot_location in location.spawns and location.is_open:
                            # print("\tPassed over a waterable plant")
                            reward -= 0.25
                            break
            return reward, False

        # move up to water
        elif action == 1:
            for dist, plant in plants_by_distance.items():
                if dist > close_radius:
                    continue
                if plant.category == "plant_good":
                    if not self.watered[plant.name]:
                        self.watered[plant.name] = True
                        reward += 2.0
                elif plant.category == "plant_evil":
                    print(
                        "🌶️ Tried to water an evil plant. "
                        f"Terminated in {self.step_number} steps "
                        f"with watered fraction {self.watered_plant_fraction()}."
                    )
                    self.is_dead = True
                    return -5.0, True
                else:
                    raise RuntimeError(f"Unknown category {plant.category}")
            if reward == 0.0:  # nothing watered, wasted water
                # print("\tWasted water")
                reward = -0.5

        # Reward shaping to get the robot to visit the charger when battery is low,
        # but not when it is high.
        if action == 2:
            if self.battery_level() < 5.0:
                # print("\tCharged when battery low :)")
                reward += 1.0
            else:
                # print(f"\tCharged when battery high ({self.battery_level()}) :(")
                reward -= 0.5

        # print(f"{self.watered=}")
        terminated = all(self.watered.values())
        if terminated:
            print(f"💧 Watered all good plants! Succeeded in {self.step_number} steps.")

        return reward, terminated

    def dead(self):
        return self.is_dead

    def watered_plant_fraction(self):
        n_watered = 0
        for w in self.watered.values():
            if w:
                n_watered += 1
        return n_watered / len(self.watered)

    def battery_level(self):
        robot_state = self.world_state.robots[0]
        return robot_state.battery_level

    def _get_plants_by_distance(self, world_state: WorldState):
        robot_state = world_state.robots[0]
        robot_pos = robot_state.pose.position
        # print(robot_pos)

        plants_by_distance = {}
        for obj in world_state.objects:
            pos = obj.pose.position
            dist = _dist(robot_pos, pos)
            dist = min(dist, self.max_dist)
            plants_by_distance[dist] = obj

        return plants_by_distance

    def reset(self, seed=None, options=None):
        super().reset(seed)

        valid_reset = False
        num_reset_attempts = 0
        while not valid_reset:
            future = self.reset_world_client.call_async(
                ResetWorld.Request(seed=(seed or -1))
            )
            rclpy.spin_until_future_complete(self.node, future)

            # Validate that there are no two plants in the same location.
            observation = self._get_obs()
            valid_reset = True
            parent_locs = set()
            for obj in self.world_state.objects:
                if obj.parent not in parent_locs:
                    parent_locs.add(obj.parent)
                else:
                    valid_reset = False
                    break

            num_reset_attempts += 1
            seed = None  # subsequent resets need to not use a fixed seed

        # Reset helper vars
        self.step_number = 0
        self.waypoint_i = -1
        self.watered = {plant: False for plant in self.good_plants}
        self.go_to_next_wp()

        print(f"Reset environment in {num_reset_attempts} attempt(s).")
        return observation, {}

    def _get_obs(self):
        """Calculate the observations"""
        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        world_state = future.result().state
        plants_by_distance = self._get_plants_by_distance(world_state)

        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[:] = -1  # unknown class
        n_observations = 0
        while n_observations < self.max_n_objects:
            closest_d = min(plants_by_distance.keys())
            plant = plants_by_distance.pop(closest_d)
            plant_class = 0 if plant.category == "plant_good" else 1
            obs[2 * n_observations] = plant_class
            obs[2 * n_observations + 1] = closest_d
            n_observations += 1

        obs[-2] = self.battery_level() / 100.0
        obs[-1] = 0.0
        cur_loc = world_state.robots[0].last_visited_location
        for loc in world_state.locations:
            if cur_loc == loc.name or cur_loc in loc.spawns:
                if not loc.is_open:
                    obs[-1] = 1.0  # closed = watered
                    break

        self.world_state = world_state
        return obs

    def eval(self):
        """Return values of custom metrics for evaluation."""
        return {
            "watered_plant_fraction": float(self.watered_plant_fraction()),
            "battery_level": float(self.battery_level()),
        }

    def get_next_navigation_action(self):
        self.waypoint_i = (self.waypoint_i + 1) % len(self.waypoints)
        return self.get_current_navigation_action()

    def get_current_navigation_action(self):
        return TaskAction(type="navigate", target_location=self.get_current_location())

    def get_current_location(self):
        return self.waypoints[self.waypoint_i]

    def go_to_next_wp(self):
        nav_goal = ExecuteTaskAction.Goal()
        nav_goal.action = self.get_next_navigation_action()
        nav_goal.action.robot = "robot"
        nav_goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

    def go_to_charger(self):
        nav_goal = ExecuteTaskAction.Goal()
        nav_goal.action = TaskAction(type="navigate", target_location="charger")
        nav_goal.action.robot = "robot"
        nav_goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
