import numpy as np
import torch

from utils import rollout, discount_rewards

def train_with_ppo(model, trainer, env, params, data_modifier_hook=None):
  device = model.get_device()
  ep_rewards = []
  for episode_idx in range(params['n_episodes']):
    # Perform rollout
    train_data, reward = rollout(model, env)
    ep_rewards.append(reward)
    
    if data_modifier_hook:
      train_data = data_modifier_hook(train_data)

    # Shuffle data
    permute_idxs = np.random.permutation(len(train_data[1]))

    obs = torch.tensor(train_data[0][:-1][permute_idxs],
                       dtype=torch.float32, device=device)
    acts = torch.tensor(train_data[1][permute_idxs],
                        dtype=torch.int32, device=device)
    gaes = torch.tensor(train_data[3][permute_idxs],
                        dtype=torch.float32, device=device)
    act_log_probs = torch.tensor(train_data[4][permute_idxs],
                                 dtype=torch.float32, device=device)

    returns = discount_rewards(train_data[2])[permute_idxs]
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # Train model
    trainer.train_policy(obs, acts, act_log_probs, gaes)
    trainer.train_value(obs, returns)

    if 'print_freq' in params and (episode_idx + 1) % params['print_freq'] == 0:
      print('Episode {} | Avg Reward {:.1f}'.format(
          episode_idx + 1, np.mean(ep_rewards[-params['print_freq']:])))
      
  return ep_rewards