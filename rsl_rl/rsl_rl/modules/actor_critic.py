# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import typing as tp

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributions import Normal

ActivationType = tp.Literal["elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid"]


@dataclass
class PolicyConfig:
    activation: ActivationType = "elu"
    actor_hidden_dims: tp.List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: tp.List[int] = field(default_factory=lambda: [512, 256, 128])
    init_noise_std: float = 1.0


def get_activation(act_name: ActivationType) -> nn.Module:
    return {"elu": nn.ELU(),
            "selu": nn.SELU(),
            "relu": nn.ReLU(),
            "crelu": nn.ReLU(), 
            "lrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
    }[act_name]


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self, 
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 policy_config: PolicyConfig = PolicyConfig(),
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        self.policy_cfg = policy_config

        activation = get_activation(policy_config.activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, self.policy_cfg.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(self.policy_cfg.actor_hidden_dims)):
            if l == len(self.policy_cfg.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(self.policy_cfg.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(self.policy_cfg.actor_hidden_dims[l], self.policy_cfg.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, self.policy_cfg.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(self.policy_cfg.critic_hidden_dims)):
            if l == len(self.policy_cfg.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(self.policy_cfg.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(self.policy_cfg.critic_hidden_dims[l], self.policy_cfg.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(self.policy_cfg.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential: tp.List[nn.Module], scales: tp.List[float]):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones: torch.Tensor | None = None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev
    
    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        value = self.critic(critic_observations)
        return value


class ActorCriticNostd(ActorCritic):
    def __init__(self,
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 actor_hidden_dims: tp.List[int] = [256, 256, 256],
                 critic_hidden_dims: tp.List[int] = [256, 256, 256],
                 activation: ActivationType = 'elu',
                 init_noise_std: float = 1,
                 **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims, critic_hidden_dims,
                         activation, init_noise_std, **kwargs)
        self.action: torch.Tensor | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self.action

    def update_distribution(self, observations: torch.Tensor):
        self.actor.train()
        self.action = self.actor(observations)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.action_mean
