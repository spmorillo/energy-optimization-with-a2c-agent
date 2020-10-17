"""
VERSION 2: JUST ONE SELECTION PER TIMESTEP

Things to try:
Sample different simulation times and UEs?
It might be more robust but env becomes stochastic
Update every 2, 5, 10... steps
Different hidden size, lr, and decays
Undiscounted and average rewards
Rewards as percentage from state_n to state_n+1
MIRAR PROPAGACIÓN DEL GRADIENTE!!
"""

import subprocess
import os
import csv
import datetime
from time import time
from operator import truediv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from resource_manager.rl_agent import RLAgent

# RL training constants
EPISODES = 1
PARALLEL_ENV = 2
GAMMA = 0.99
AVAILABILITY_PENALTY = 0
ENTROPY_PENALTY = 1e-3

# Paths to files
MERCURY_PATH = "main_heterogeneous.py" #"scenarios/main_heterogeneous.py"
MODEL_FOLDER = "rl/training/a2c_tfm/"

# Other constants
EPSILON = np.finfo(np.float32).eps
COLUMN_NAMES = ["util_0", "power_0", "it_0", "cool_0",
                "util_1", "power_1", "it_1", "cool_1",
                "util_2", "power_2", "it_2", "cool_2",
                "action", "penalty"]
SCALED_PAIR = {"power": (944.16, 183.39),
               "util": (656.54, 222.56)}

class MercuryEnv:
    def __init__(self):
        self.index = 0
        self.obs = []
        self.actions = []
        self.penalties = []
        self.offset = 0

    def reset(self, path):
        # Removing first value to avoid noise (Removed)
        data = pd.read_csv(path, delimiter=';', names=COLUMN_NAMES)
        self.index = 0
        self.obs = data.iloc[:, :-2].astype(np.float32)
        label_encoder = LabelEncoder()
        label_encoder.fit(["edc_0", "edc_1", "edc_2"])
        self.actions = label_encoder.transform(data["action"].values)
        self.penalties = data["penalty"].values.astype(np.bool)
        #self.offset = np.sum(np.mean(self.obs[:, 1::4], axis=0))/self.obs[:, 1::4].shape[1]
        return self.obs.iloc[self.index].values, False, self.actions[self.index]

    def step(self):
        self.index += 1
        obs = self.obs.iloc[self.index].values
        action = self.actions[self.index]
        done = self.index == self.obs.shape[0]-1
        reward = 0 if done else self.compute_reward()
        return obs, reward, done, action

    def compute_reward(self):
        power_consumption = self.inverse_transform(SCALED_PAIR["power"])
        total_power = np.sum(power_consumption)
        penalty = self.penalties[self.index]
        if total_power > 0:
            return -(1 + AVAILABILITY_PENALTY * penalty) * total_power
        return -(1 - AVAILABILITY_PENALTY * penalty) * total_power

    def inverse_transform(self, scaled_pair):
        scaled_data = self.obs.loc[self.index+1, ["power_0","power_1","power_2"]].values # Total power
        return scaled_data * scaled_pair[1] + scaled_pair[0]

def main():

    # Initialize environment and agent object, get max rewards
    env = MercuryEnv()
    actor_critic = RLAgent(MODEL_FOLDER)
    max_reward = actor_critic.get_max_reward()

    for episode in range(EPISODES):

        # Remove old data, call several Mercury sim to collect new, reset env to use them
        start_time = time()
        parallel_envs = []
        for env_id in range(PARALLEL_ENV):
            data_path = (MODEL_FOLDER+"current_trajectories/n_{}.csv").format(env_id)
            if os.path.exists(data_path):
                subprocess.call(['rm', data_path])
            parallel_env = subprocess.Popen(['python', MERCURY_PATH, MODEL_FOLDER, str(env_id)])
            parallel_envs.append(parallel_env)
        for parallel_env in parallel_envs:
            parallel_env.wait()
        end_time = time()

        # For each dataset generated by an env compute Vvals, Qvals log_prob, rewards and concatenated them
        log_probs = []
        rewards = []
        v_vals = []
        q_vals = []
        entropies = []

        for env_id in range(PARALLEL_ENV):

            state, done, action = env.reset(path=(MODEL_FOLDER+"current_trajectories/n_{}.csv").format(env_id))

            env_log_probs = []
            env_rewards = []
            env_v_vals = []
            env_entropies = []

            while not done:

                value, policy_dist = actor_critic.forward(state)
                v_val = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy()

                # Action is sampled already in Mercury
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(dist * np.log(dist))
                new_state, reward, done, action = env.step()

                env_log_probs.append(log_prob)
                env_rewards.append(reward)
                env_v_vals.append(v_val)
                env_entropies.append(entropy)
                state = new_state

                with open(MODEL_FOLDER+"info/dist.txt", 'a') as f:
                    f.write(','.join([str(elem) for elem in dist]))
                    f.write('\n')

            # Removing last interaction whose reward is unknown
            env_log_probs.pop()
            env_rewards.pop()
            env_v_vals.pop()
            env_entropies.pop()

            # Compute Q values
            env_rewards_scaled = (env_rewards - np.mean(env_rewards)) / (np.std(env_rewards) + EPSILON)
            q_val = env_rewards_scaled[-1]
            env_q_vals = np.zeros_like(env_v_vals)
            for i in reversed(range(len(env_rewards_scaled))):
                q_val = env_rewards_scaled[i] + GAMMA * q_val
                env_q_vals[i] = q_val
            env_q_vals = (env_q_vals - np.mean(env_q_vals)) / (np.std(env_q_vals) + EPSILON)

            log_probs.extend(env_log_probs)
            rewards.extend(env_rewards)
            v_vals.extend(env_v_vals)
            q_vals.extend(env_q_vals)
            entropies.extend(env_entropies)

            with open(MODEL_FOLDER+"info/train_arrays.txt", 'a') as f:
                for a, b, c, d, e in zip(env_log_probs, env_rewards, env_v_vals, env_q_vals, env_entropies):
                    f.write(','.join([str(elem) for elem in [a, b, c, d, e]]))
                    f.write('\n')

        # Update actor-critic
        log_probs = torch.stack(log_probs)
        v_vals = torch.FloatTensor(v_vals)
        q_vals = torch.FloatTensor(q_vals)
        entropies = torch.FloatTensor(entropies)

        a_vals = q_vals - v_vals
        actor_loss = (-log_probs * a_vals).mean()
        critic_loss = 0.5 * a_vals.pow(2).mean()
        entropy_term = -ENTROPY_PENALTY * entropies.mean()
        ac_loss = actor_loss + critic_loss + entropy_term

        actor_critic.ac_optimizer.zero_grad()
        ac_loss.backward()
        actor_critic.ac_optimizer.step()

        #Save model weights each episode and store best model
        if np.sum(rewards)/len(rewards) > max_reward:
            max_reward = np.sum(rewards)/len(rewards)
            actor_critic.update_weights("model_weights/best.pt", max_reward)
            with open(MODEL_FOLDER+"info/info.txt", 'a') as info_file:
                info_file.write("Saving new best model's weigth! \
                                 Episode: {},  Reward: {}\n".format(episode, max_reward))
        actor_critic.update_weights("model_weights/current.pt", max_reward)

        #Save info about the episode
        with open(MODEL_FOLDER+"info/episodes.csv", 'a') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([episode, end_time-start_time,
                            len(rewards), np.sum(rewards), 
                            ac_loss.detach().numpy().item(),
                            actor_loss.detach().numpy().item(),
                            critic_loss.detach().numpy().item(),
                            entropy_term.detach().numpy().item()])

if __name__ == "__main__":
    start = time()
    main()
    print('****************************************')
    print('Final elongated time: {}'.format(datetime.timedelta(seconds=time()-start)))
    print('****************************************')
    plt.plot([0])
    plt.show()