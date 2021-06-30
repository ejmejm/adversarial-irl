from collections import namedtuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical

def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

TrajectoryData = namedtuple(
  'TrajectoryData',
  ['obs',
   'acts',
   'rewards',
   'baselines',
   'act_log_probs'])
  
def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    device = model.get_device()
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs = env.reset()
    
    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor([obs], dtype=torch.float32,
                                         device=device))
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, _ = env.step(act)
        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
          train_data[i].append(item)
        obs = next_obs
        ep_reward += reward
        if done:
            break

    train_data[0].append(obs)
    train_data = [np.asarray(x) for x in train_data]
    # Calculate GAEs and replace values with GAE values.
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    
    train_data = TrajectoryData(*train_data)

    return train_data, ep_reward