import torch
from torch import nn

# For 1D observation spaces and discrete action spaces
class FCActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size, hidden_size=64):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU())
    
    self.policy_layers = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_space_size))
    
    self.value_layers = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1))
    
  def get_device(self):
    return next(self.parameters()).device
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value


# For 1D observation spaces and discrete action spaces
class FCReturnModel(nn.Module):
  def __init__(self, obs_space_size, action_space_size=0, hidden_size=64):
    super().__init__()

    self.obs_space_size = obs_space_size
    self.action_space_size = action_space_size
    self.use_actions = action_space_size > 0
    
    self.obs_layers = nn.Sequential(
        nn.Linear(obs_space_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU())
    
    # Add layers for action if using discriminator is dependent on actions
    if self.use_actions:
      self.act_layers = nn.Sequential(
          nn.Linear(action_space_size, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, hidden_size),
          nn.ReLU())
      reward_in_hidden = 2 * hidden_size
    else:
      reward_in_hidden = hidden_size
    
    self.reward_layers = nn.Sequential(
        nn.Linear(reward_in_hidden, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1))
    
    self.value_layers = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1))
    
  def get_device(self):
    return next(self.parameters()).device
    
  def reward(self, obs, act=None):
    z = self.obs_layers(obs)
    if self.use_actions:
      act_hidden = self.act_layers(act)
      z = torch.cat([z, act_hidden], dim=1)
    reward = self.reward_layers(z)
    return reward
  
  def value(self, obs):
    z = self.obs_layers(obs)
    value = self.value_layers(z)
    return value

  def forward(self, obs, act=None):
    obs_hidden = self.obs_layers(obs)
    if self.use_actions:
      act_hidden = self.act_layers(act)
      full_hidden = torch.cat([obs_hidden, act_hidden], dim=1)
      reward = self.reward_layers(full_hidden)
    else:
      reward = self.reward_layers(obs_hidden)
    value = self.value_layers(obs_hidden)
    return reward, value
  
  def discrim(self, obs, next_obs, act_probs, acts=None, gamma=0.99):
    rewards, values = self.forward(obs, acts)
    next_values = self.value(next_obs)
    advantages = rewards + gamma * next_values - values
    exp_adv = torch.exp(advantages)
    out = exp_adv / (exp_adv + act_probs)
    return out

if __name__ == '__main__':
  ac = FCActorCriticNetwork(5, 5)
  print(ac.get_device())
  
  rm = FCReturnModel(4, 0)
  obs = torch.rand((2, 4))
  act_probs = torch.rand((2, 1))
  print(rm.get_device())
  print(rm.reward(obs).shape)
  print(rm.value(obs).shape)
  print(*(x.shape for x in rm.forward(obs)))
  print(rm.discrim(obs, obs, act_probs).shape)
  
  rm = FCReturnModel(4, 3)
  act = torch.rand((2, 3))
  print(rm.get_device())
  print(rm.reward(obs, act).shape)
  print(rm.value(obs).shape)
  print(*(x.shape for x in rm.forward(obs, act)))
  print(rm.discrim(obs, obs, act_probs, act).shape)