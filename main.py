import gym
import numpy as np
import torch

from models import FCActorCriticNetwork
from trainers import PPOTrainer
from training import train
from utils import rollout, discount_rewards

DEVICE = 'cuda'

def ppo_training_setup(env):
  # Object instantiation
  model = FCActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
  model = model.to(DEVICE)

  # Define training params
  params = {
    'n_episodes': 200,
    'print_freq': 20
  }
  
  ppo = PPOTrainer(
      model,
      policy_lr = 3e-4,
      value_lr = 1e-3,
      target_kl_div = 0.02,
      max_policy_train_iters = 40,
      value_train_iters = 40)
  
  return model, ppo, params
  
if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  model, ppo_trainer, params = ppo_training_setup(env)
  rewards = train(model, ppo_trainer, env, params)
  print(rewards)