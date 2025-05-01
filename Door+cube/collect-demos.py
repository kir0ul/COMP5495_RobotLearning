# Script adapted from examples at https://robosuite.ai/docs/index.html

import argparse
import os
import random
import shutil
import time

import h5py
import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper
from glob import glob
from scipy import interpolate
from scipy.interpolate import interp1d

from door_custom import DoorCustom


def write_to_h5(
    interp, timesteps, timestamps, states, actions, demNum, ep_directory, max_fr
):
    print("Writing demo data to h5 file")
    if interp:
        t = np.arange(0, timesteps)
        inttimestamps = interp1d(t, timestamps)
        # intstates = interp1d(t, states)
        # intactions = interp1d(t, actions)

    grp = f.create_group("demo" + str(demNum))
    grp.create_dataset("timestamps", data=np.array(timestamps))
    grp.create_dataset("states", data=np.array(states))
    grp.create_dataset("actions", data=np.array(actions))
    grp.create_dataset("xmlmodel", data=ep_directory + "/model.xml")
    with open(ep_directory + "/model.xml", "r") as xml:
        grp.attrs["xmlmodel"] = xml.read()
    grp.attrs["episode"] = ep_directory
    grp.attrs["max_fr"] = max_fr


def playback_trajectory(env, ep_dir, max_fr=None):
    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            start = time.time()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.viewer.update()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)

            if max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
    env.close()


if __name__ == "__main__":
    timesteps = 700
    interp = 0
    discardedDems = np.zeros(0)
    numDiscards = 0
    seed = random.randint(1, 10000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Door")
    parser.add_argument("--directory", type=str, default=os.getcwd())
    # parser.add_argument("--mode", type=str, default="record")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="UR5e",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--switch-on-grasp",
        action="store_true",
        help="Switch gripper control on gripper action",
    )
    parser.add_argument(
        "--toggle-camera-on-grasp",
        action="store_true",
        help="Switch camera angle on gripper action",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        horizon=100,
        # seed=seed,
    )
    # env.seed = seed
    # interp = input("Perform Smoothing? y/n")
    # if(interp == 'y'):
    # interp = 1
    data_directory = args.directory
    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)
    env = DataCollectionWrapper(env, data_directory)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard', 'dualsene' or 'spacemouse'."
        )

    prefix = "ep_"
    for filename in os.listdir(os.getcwd()):
        if filename.startswith(prefix):
            shutil.rmtree(os.getcwd() + "/" + filename)

    extension = ".h5"
    for filename in os.listdir():
        if filename.endswith(extension):
            os.remove(filename)

    demNum = 0
    continueDems = "y"
    f = h5py.File("demos.h5", "w")
    while continueDems == "y":
        task_completion_hold_count = (
            -1
        )  # counter to collect 10 timesteps after reaching goal

        demNum = demNum + 1
        # Reset the environment

        obs = env.reset()
        # env.seed=seed
        #
        #     # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]
        start_time = time.time()
        done = False

        states = []
        actions = []
        timestamps = []
        # Loop until we get a reset from the input or the task completes
        for x in range(timesteps):
            if x == 99:
                pass
            if x % 100 == 0:
                print(x)
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            from copy import deepcopy

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(
                    active_robot.composite_controller, WholeBody
                ):  # input type passed to joint_action_policy
                    controller_input_type = (
                        active_robot.composite_controller.joint_action_policy.input_type
                    )
                else:
                    controller_input_type = active_robot.part_controllers[
                        arm
                    ].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i])
                for i, robot in enumerate(env.robots)
            ]
            env_action[device.active_robot] = active_robot.create_action_vector(
                action_dict
            )
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[
                    gripper_ac
                ]

            # step
            obs, reward, done, info = env.step(env_action)
            state = env.sim.get_state().flatten()  # get_state(): return MjSimState(self.data.time, qpos, qvel, act, udd_state) flatten(): return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)
            actions.append(env_action)
            states.append(state)
            timestamps.append(time.time() - start_time)
            env.render()

            # limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

            if done:
                print("Episode complete")
                running = False
                obs = env.reset()
                break

            # Also break if we complete the task
            if task_completion_hold_count == 0:
                break
            # state machine to check for having a success for 10 consecutive timesteps
            if env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = (
                        10  # reset count on first success timestep
                    )
            else:
                task_completion_hold_count = (
                    -1
                )  # null the counter if there's no success

        keepDem = input("Keep Demonstration? y/n ")
        if keepDem == "y":
            write_to_h5(
                interp,
                timesteps,
                timestamps,
                states,
                actions,
                demNum,
                env.ep_directory,
                args.max_fr,
            )

        else:
            demNum = demNum - 1
            numDiscards = numDiscards + 1
            discardedDems = np.append(discardedDems, env.ep_directory)

        continueDems = input("Continue Demonstrations? y/n ")

    # playback some data
    reviewDem = input("Review Demonstrations? y/n ")
    if reviewDem == "y":
        for filename in os.listdir(os.getcwd()):
            if filename.startswith(prefix):
                print("Reading " + filename)
                data_directory = os.getcwd() + "/" + filename
                playback_trajectory(env, data_directory, args.max_fr)

    print("reached end, closing...")
    env.close()

if numDiscards > 0:
    for i in range(numDiscards):
        temp = discardedDems[i]
        shutil.rmtree(temp)
