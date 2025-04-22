#!/usr/bin/env python3

import numpy as np
import robosuite

# from robosuite.models import MujocoWorldBase
# from robosuite.models.robots import UR5e
# from robosuite.models.grippers import gripper_factory
# from robosuite.models.arenas import TableArena
# from robosuite.models.objects import BallObject
# from robosuite.utils.mjcf_utils import new_joint
from door_custom import DoorCustom

# create environment instance
env = robosuite.make(
    env_name="DoorCustom",  # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    door_open=True,
)
# world = MujocoWorldBase()
# mujoco_robot = UR5e()
# gripper = gripper_factory('Robotiq85Gripper')
# mujoco_robot.add_gripper(gripper)
# mujoco_robot.set_base_xpos([0, 0, 0])
# world.merge(mujoco_robot)
# mujoco_arena = TableArena()
# mujoco_arena.set_origin([0.8, 0, 0])
# world.merge(mujoco_arena)


# reset the environment
env.reset()

for i in range(100):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
