This repository holds the code for my Masters Thesis about "Solving Sokoban Levels With Reinforcement Learning"

This repository is split into two subfolders, NetLogo and Python.

## NetLogo
This Folder holds the .netlogo file along with one level of Boxobans (https://github.com/google-deepmind/boxoban-levels) medium Collection.
For further Information i refer to the NetLogo-file.


## Python
The submodule 'gym_sokoban' is a forked repository from https://github.com/mpSchrader/gym-sokoban that was updated to support Gym v26 and extended with two environments.
I deleted the fork to display my changes, including the added files in the Python subfolder. For the original work by mpSchrader, please refer to his GitHub. **All credit for the base environment belongs to him!**

In the Tabular_QLearning.py-file, a QLearning-agent is defined along with the implementation to train that agent.
The DQN.py is implemented using stable-baselines3.

The libraries needed to run this are provided in the env.yaml file, which you can import using e.g. Anaconda.

Note that you might need to add the gym_sokoban subfolder to your PATH in order to run this.
