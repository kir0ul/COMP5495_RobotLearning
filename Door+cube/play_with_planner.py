"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random
import time

import h5py
import numpy as np

import robosuite
from glob import glob

import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Door")
    parser.add_argument("--directory", type=str, default="/tmp/")
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
    args = parser.parse_args()

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        horizon=100,
    )

    # create a video writer with imageio
    video_path = "video.mp4"
    video_writer = imageio.get_writer(video_path, fps=24)
    frames = []

    # location of plan
    planPath = "./planner/test.pddl.soln"
    actionLocation = "door_actions/"
    plan = open(planPath)
    actNum = 0
    for action in plan:
        currentAct = action.split(" ")[0].split("(")[1]
        actionTraj = currentAct + "_traj"

        f = h5py.File(actionLocation + actionTraj + "/" + actionTraj + ".h5", "r")
        demos = list(f.keys())

        for filename in demos:
            ep = filename
            if actNum == 0:
                xml_path = os.path.join(actionLocation, actionTraj, "model.xml")
                # xml_path = os.path.join(actionLocation, f[ep].attrs['episode'], "model.xml")
                with open(xml_path, "r") as xmlfile:
                    env.reset_from_xml_string(xmlfile.read())
            actNum = actNum + 1
            # read the model xml, using the metadata stored in the attribute for this episode
            state_paths = os.path.join(actionLocation, actionTraj, "*.npz")

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
                    if t % 10 == 0:
                        print(t)

                    if f[ep].attrs["max_fr"] is not None:
                        elapsed = time.time() - start
                        diff = 1 / f[ep].attrs["max_fr"] - elapsed
                        if diff > 0:
                            time.sleep(diff)
    env.close()
