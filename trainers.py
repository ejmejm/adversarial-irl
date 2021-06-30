import torch
from torch import optim
from torch.distributions.categorical import Categorical
from torch.nn.functional import binary_cross_entropy as bce

class PPOTrainer():
  def __init__(self,
               actor_critic,
               ppo_clip_val=0.2,
               target_kl_div=0.01,
               max_policy_train_iters=80,
               value_train_iters=80,
               policy_lr=3e-4,
               value_lr=1e-2):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.policy_layers.parameters())
    self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.value_layers.parameters())
    self.value_optim = optim.Adam(value_params, lr=value_lr)

  def train_policy(self, obs, acts, old_log_probs, gaes):
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)
      # new_selected_logits = new_logits.gather(1, acts.unsqueeze(-1))

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      clipped_ratio = policy_ratio.clamp(
          1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
      full_loss = policy_ratio * gaes
      clipped_loss = clipped_ratio * gaes
      policy_loss = -torch.min(full_loss, clipped_loss).mean()

      policy_loss.backward()
      self.policy_optim.step()

      kl_div = (old_log_probs - new_log_probs).mean()
      if kl_div >= self.target_kl_div:
        break

  def train_value(self, obs, returns):
    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()

      values = self.ac.value(obs)
      value_loss = (returns - values) ** 2
      value_loss = value_loss.mean()

      value_loss.backward()
      self.value_optim.step()
      
    
class AIRLTrainer():
  def __init__(self,
               return_model,
               reward_lr=1e-4,
#                value_lr=1e-3,
               discrim_lr=1e-4):
    self.model = return_model

    reward_params = list(self.model.obs_layers.parameters()) + \
        list(self.model.reward_layers.parameters())
    if self.model.use_actions:
      reward_params += list(self.model.act_layers.parameters())
    self.reward_optim = optim.Adam(reward_params, lr=reward_lr)

#     value_params = list(self.model.obs_layers.parameters()) + \
#         list(self.model.value_layers.parameters())
#     self.value_optim = optim.Adam(value_params, lr=value_lr)

    self.discrim_optim = optim.Adam(self.model.parameters(), lr=discrim_lr)
  
  # Label are 1 for real, 0 for fake
  # I should probably and try to balance sources
  # Also consider adding GAN label training tricks from https://github.com/ejmejm/Drawing-Improver/blob/master/model_gan.ipynb
  def train_discrim(self, obs, next_obs, act_probs, labels, acts=None, gamma=0.99):
    self.discrim_optim.zero_grad()
    
    discrim_preds = self.model.discrim(obs, next_obs, act_probs, acts=acts, gamma=gamma)
    loss = bce(discrim_preds, labels)
    
    loss.backward()
    self.discrim_optim.step()
    
    return loss.item()
    
#   def train_reward(self, obs, next_obs, act_probs, act=None, gamma=0.99):
#     self.reward_optim.zero_grad()
    
#     with torch.no_grad():
#       discrim_preds = self.model.discrim(obs, next_obs, act_probs, acts=acts, gamma=gamma)
    
#     target_rewards = torch.log(discrim_preds) - torch.log(1.0 - discrim_preds)
#     pred_rewards = self.model.reward(obs, act)
#     loss = (target_rewards - pred_rewards) ** 2
    
#     loss.backward()
#     self.reward_optim.step()
    
#     return loss.item()