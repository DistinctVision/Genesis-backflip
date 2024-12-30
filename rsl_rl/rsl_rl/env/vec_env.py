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
from abc import ABC, abstractmethod
import torch

# minimal interface of the environment
class VecEnv(ABC):

    def __init__(self,
                 num_envs: int,
                 num_obs: int,
                 num_privileged_obs: int,
                 num_actions: int,
                 max_episode_length: int,
                 privileged_obs_buf: torch.Tensor | None = None,
                 obs_buf: torch.Tensor | None = None,
                 rew_buf: torch.Tensor | None = None,
                 reset_buf: torch.Tensor | None = None,
                 episode_length_buf: torch.Tensor | None = None, # current episode duration
                 extras: tp.Dict[str, tp.Any] = {},
                 device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_actions = num_actions
        self.max_episode_length = max_episode_length
        self.privileged_obs_buf = privileged_obs_buf
        self.obs_buf = obs_buf
        self.rew_buf = rew_buf
        self.reset_buf = reset_buf
        self.episode_length_buf = episode_length_buf
        self.extras = extras
        self.device = torch.device(device)

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        ...
    
    @abstractmethod
    def reset(self, env_ids: tp.Union[list, torch.Tensor]):
        ...
    
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        ...
    
    @abstractmethod
    def get_privileged_observations(self) -> tp.Union[torch.Tensor, None]:
        ...