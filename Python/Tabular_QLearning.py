import gymnasium as gym

import time
from PIL import Image
import numpy as np
import argparse
import os

from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

from gym_sokoban.gym_sokoban.envs import SokobanEnv, SokobanEnvQLearningSameLevel

import csv

import multiprocessing


learning_rates = [0.5]
disc_factors = [0.8]

n_episodes = 100000

initial_epsilon = 1.0
final_epsilon = 0.01
decay_rate = 0.0005


def learn(lr, dfactor):
    env = gym.make("Sokoban-qLearning-v0", render_mode="human")

    class QLearningAgent:
        def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            final_epsilon: float,
            discount_factor: float,
            decay_rate: float,
        ):
            self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))

            self.lr = learning_rate
            self.discount_factor = discount_factor

            self.epsilon = initial_epsilon
            self.initial_epsilon = initial_epsilon
            self.final_epsilon = final_epsilon
            self.decay_rate = decay_rate

            self.training_error = []

        def get_action(self, obs: tuple[np.int32, np.int32, np.int32]) -> int:
            if np.random.random() < self.epsilon:
                return env.action_space.sample()
            
            else:
                return int(np.argmax(self.qValues[tuple(obs)]))
            
        def update(
            self,
            obs: tuple[np.int32, np.int32, np.int32],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[np.int32, np.int32, np.int32],
        ):
            future_q_value = np.max(self.qValues[tuple(next_obs)])
            temporal_difference = (
                reward + self.discount_factor * future_q_value - self.qValues[tuple(obs)][action]
            )

            self.qValues[tuple(obs)][action] = (
                self.qValues[tuple(obs)][action] + self.lr * temporal_difference
            )
            self.training_error.append(temporal_difference)

        def decay_epsilon(self, episode):
            # self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
            self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * np.exp(-self.decay_rate * episode)

        def get_q_table_length(self):
            return len(self.qValues)
        
        def print_q_table(self):
            for key, value in self.qValues.items():
                print(f"State: {key}, Q-Values: {value}")

        def get_epsilon(self):
            return self.epsilon

    agent = QLearningAgent(learning_rate= lr,
                        initial_epsilon=initial_epsilon,
                        final_epsilon=final_epsilon,
                        discount_factor=dfactor,
                        decay_rate=decay_rate
                        )

    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    filename = "logs.csv"
    print(filename)
    columns = ["epoch", "meanReward", "times-finished", "time-last-epoch", "deadlock-cnt", "no-op-count", "qtable-rows", "current_epsilon"]
    start_time = time.perf_counter()
    reward_epoch = 0
    with open(filename, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False
            total_reward = 0
            if episode % 100 == 1:
                start_time = time.perf_counter()
                reward_epoch = 0
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs
                total_reward += reward

            agent.decay_epsilon(episode)
            reward_epoch += total_reward
            if episode % 100 == 0:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                data = {"epoch": episode/100, 
                        "meanReward": reward_epoch/100,
                        "times-finished": info["done_cnt"],
                        "time-last-epoch": elapsed,
                        "deadlock-cnt": info["dead_end_cnt"],
                        "no-op-count": info["no_op_cnt"],
                        "qtable-rows": agent.get_q_table_length(),
                        "current_epsilon": agent.get_epsilon() }
                writer.writerow(data)


def start_process(lr, df):
    learn(lr, df)

if __name__=="__main__":
    processes = []
    for lr in learning_rates:
        for dfactor in disc_factors:
            process = multiprocessing.Process(target=start_process, args=(lr, dfactor))
            processes.append(process)
            process.start() 

    for process in processes:
        process.join()
