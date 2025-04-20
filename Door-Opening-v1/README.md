# README

Here are the main entrypoint files of this project:

- `main.py`: This script allows for the collection of demonstration data for a given environment. The timesteps variable controls how long the simulation runs for. The collected data is written to an `.h5` file. The script also allows during runtime for demonstrations to be discarded or reviewed and for the user to choose how many times to demonstrate.

- `playback_from_h5.py`: This script takes an h5 file and plays back a simulation of the robot's movement in robosuite.

- `ProMP.py`: Using an h5 file, this script will generate a ProMP trajectory and write it to an `.h5` file.

- `play_with_planner/play_with_planner.py`: This script takes a `.soln` plan and a folder of possible actions and plays a simulation of the plan.
