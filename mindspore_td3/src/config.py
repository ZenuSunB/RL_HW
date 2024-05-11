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
TD3 config.
"""
import mindspore as ms
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment

from .td3 import TD3Actor, TD3Learner, TD3Policy
collect_env_params = {'name': 'FightingiceEnv-v0'}
eval_env_params = {'name': 'FightingiceEnv-v0'}


policy_params = {
    "state_space_dim": 144,
    "action_space_dim": 40,
    "hidden_size1": 512,
    "hidden_size2": 512,
    "compute_type": ms.float32,
}

learner_params = {
    "gamma": 0.995,
    "state_space_dim": 144,
    "action_space_dim": 40,
    "actor_lr": 1e-3,
    "critic_lr": 1e-4,
    "actor_update_interval": 2,
    "action_boundary": [0, 1.0],
    "target_update_factor": 0.05,
    "target_update_interval": 5,
    "target_action_noise_stddev": 0.1,
    "target_action_noise_clip": 0.2,
}

trainer_params = {
    "ckpt_path": "./ckpt",
    "num_eval_episode": 1,
    "save_per_episode": 20,
    'buffer_num_before_learning_begin': 32,
    'update_target_iter': 100,
    "init_collect_size" : 32,
}

actor_params = {"actor_explore_noise": 0.1}

algorithm_config = {
    "actor": {
        "number": 1,
        "type": TD3Actor,
        "params": actor_params,
        "networks": ["actor_net"],
        "policies": ['init_policy', 'evaluate_policy', 'collect_policy' ],
    },
    "learner": {
        "number": 1,
        "type": TD3Learner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "target_actor_net",
            "critic_net_1",
            "critic_net_2",
            "target_critic_net_1",
            "target_critic_net_2",
        ],
    },
    "policy_and_network": {"type": TD3Policy, "params": policy_params},
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': collect_env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {'number': 1,
                      'type': UniformReplayBuffer,
                      'capacity': 20000,
                      'data_shape': [(144,), (40,), (1,), (144,), (1,)],
                      'data_type': [ms.float32, ms.float32, ms.float32, ms.float32, ms.bool_],
                      'sample_size': 32},
}

summary_config = {
    "mindinsight_on": False,
    "base_dir": "./summary",
    "collect_interval": 2,
}
