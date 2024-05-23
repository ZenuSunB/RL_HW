# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
SAC config.
"""
import mindspore
from .sac import SACActor, SACLearner, SACPolicy
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment
collect_env_params = {'name': 'FightingiceEnv-v0'}
eval_env_params = {'name': 'FightingiceEnv-v0'}
policy_params = {
    "state_space_dim": 144,
    "action_space_dim": 40,
    "hidden_size": 512,
}

learner_params = {
    "gamma": 0.90,
    "state_space_dim": 144,
    "action_space_dim": 40,
    "epsilon": 0.2,
    "critic_lr": 3e-4,
    "actor_lr": 3e-4,
    "log_alpha": 0.0,
    "train_alpha_net": True,
    "alpha_lr": 3e-4,
    "target_entropy": -3.0,
    "update_factor": 0.005,
    "update_interval": 1,
}

trainer_params = {
    "duration": 2,
    "batch_size": 32,
    "save_per_episode": 100,
    "ckpt_path": "./ckpt",
    "num_eval_episode": 2,
}

algorithm_config = {
    "replay_buffer": {
        "number": 1,
        "type": UniformReplayBuffer,
        "capacity": 20000,
        "sample_size": 32,
        'data_shape': [(144,), (1,), (1,), (144,), (1,)],
        'data_type': [mindspore.float32, mindspore.int32, mindspore.float32, mindspore.float32, mindspore.bool_],
    },
    "actor": {
        "number": 1,
        "type": SACActor,
        "policies": ["init_policy", "collect_policy", "eval_policy"],
    },
    "learner": {
        "number": 1,
        "type": SACLearner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "critic_net1",
            "critic_net2",
            "target_critic_net1",
            "target_critic_net2",
        ],
    },
    "policy_and_network": {"type": SACPolicy, "params": policy_params},
    "collect_environment": {
        "number": 1,
        "type": GymEnvironment,
        "params": collect_env_params,
    },
    "eval_environment": {
        "number": 1,
        "type": GymEnvironment,
        "params": eval_env_params,
    },
}
