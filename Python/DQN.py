import sys
import gymnasium as gym

from gym_sokoban.envs.sokoban_env import SokobanEnv

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import torch
import time
from PIL import Image
import numpy as np
import argparse
import os

    
class CustomCallback(BaseCallback):
    """
    Custom callback for publishing variables to TensorBoard before each environment reset.
    """

    def __init__(self, verbose=0):
        """
        Initialize the callback.

        :param log_dir: (str) Directory to save TensorBoard logs
        :param verbose: (int) Verbosity level
        """
        self.episodes_without_finishing = 0
        self.times_finished_last = 0
        self.times_finished_now = 0


        self._called = False
        self.n_calls = 0

        self.check_freq = 30000 # approx. 100 episodes
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the parent class after each step in the environment.

        :return: (bool) Whether or not training should continue
        """
        self.times_finished_now = self.training_env.buf_infos[-1]["done_cnt"]

        # Access the custom variable from the environment
        self.logger.record("customs/no op", self.training_env.buf_infos[-1]["no_op_cnt"])
        # self.logger.record("customs/moved left", self.training_env.buf_infos[-1]["moved_left_cnt"])
        # self.logger.record("customs/moved right", self.training_env.buf_infos[-1]["moved_right_cnt"])
        # self.logger.record("customs/moved up", self.training_env.buf_infos[-1]["moved_up_cnt"])
        # self.logger.record("customs/moved down", self.training_env.buf_infos[-1]["moved_down_cnt"])
        # self.logger.record("customs/player moved", self.training_env.buf_infos[-1]["player_moved_cnt"])
        self.logger.record("customs/pushed box", self.training_env.buf_infos[-1]["pushed_box_cnt"])
        self.logger.record("customs/times done", self.times_finished_now)
        self.logger.record("customs/times done last", self.times_finished_last)
        self.logger.record("customs/current reward", self.training_env.buf_infos[-1]["current reward"]) 
        self.logger.record("customs/dead ends reached", self.training_env.buf_infos[-1]["dead_end_cnt"]) 
        self.logger.record("customs/didnt_move", self.training_env.buf_infos[-1]["didnt_move_cnt"])
                
        return True

    def _on_rollout_end(self) -> None:
        """
        This method will be called by the parent class after each environment rollout (before reset).

        :return: None
        """
        # Perform actions needed before environment reset
        # Publish variables to TensorBoard
        # For example:

        self.logger.record("reward last episode", self.training_env.buf_infos[-1]["reward_last_episode"])
        pass

MAX_STEPS = 500
TOTAL_TIMESTEPS = int(1e7)
env_name = "Sokoban-sameLevel-v0"

eval_env = Monitor(gym.make(env_name, render_mode="greyscale"))

env = gym.make(env_name, render_mode="greyscale")
obs = env.reset()
print("observation", obs)

time.sleep(4)
callback = CustomCallback()
print(env.observation_space.shape)
print(env.action_space.n)

gamma = [0.99, 0.95, 0.9, 0.85]
learning_rate = [0.0005, 0.0001, 0.0008, 0.001]
exploration_fraction = [0.1, 0.2, 0.3, 0.5]


model_path = "models"
log_dir = "logs"


eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_path}/best_model",
                    log_path="./eval_logs/results", eval_freq=10000)

callbacks = CallbackList([callback, eval_callback])
callback.n_calls = 0
# g = gamma[i]
# lr = learning_rate[i%4]
# ef = exploration_fraction[i]'
learning_rates = [0.00005, 0.00008]
for lr in learning_rates:
    model_path = "models/lr" + str(lr)
    log_dir = "logs/lr" + str(lr)

    model = DQN("MlpPolicy", env, verbose=1, 
            tensorboard_log=log_dir,
            # tensorboard_log="tmp",
            buffer_size=1000000,
            # exploration_initial_eps=0.9,
            # exploration_final_eps=0.05,
            exploration_fraction=0.6,
            learning_starts=10000,
            learning_rate=lr,
            gamma=0.99)
        # learning_starts=1e5)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10, callback=callbacks) 

    model.save(model_path)
print("done")