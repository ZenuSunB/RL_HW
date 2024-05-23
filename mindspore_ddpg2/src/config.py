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
DDPG config.
"""
import mindspore as ms
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from mindspore_rl.environment import GymEnvironment

from .ddpg import DDPGActor, DDPGLearner, DDPGPolicy
collect_env_params = {'name': 'FightingiceEnv-v0'}
eval_env_params = {'name': 'FightingiceEnv-v0'}


policy_params = {
    "state_space_dim": 144, # 状态空间维度
    "action_space_dim": 40, # 动作空间维度
    "hidden_size1": 512, # 两个隐藏层
    "hidden_size2": 512,
    "compute_type": ms.float32,
}

learner_params = {
    "gamma": 0.995, # 奖励衰减系数
    "state_space_dim": 144,
    "action_space_dim": 40,
    "actor_lr": 1e-4, # actor学习率
    "critic_lr": 1e-4, # critic学习率
    "actor_update_interval": 1, # actor网络更新间隔
    "action_boundary": [0.0, 39.9], # 动作边界
    "target_update_factor": 0.05,
    "target_update_interval": 5, # 目标网络更新间隔
    "target_action_noise_stddev": 1,
    "target_action_noise_clip": 2,
}

trainer_params = {
    "ckpt_path": "./ckpt",
    "num_eval_episode": 1, # 评估回合数
    "save_per_episode": 20, 
    'buffer_num_before_learning_begin': 32,
    'update_target_iter': 100, # 目标网络更新迭代的次数
    "init_collect_size" : 32, # 初始收集大小
    "eval_episodes": 200,
}

actor_params = {"actor_explore_noise": 0.1}

algorithm_config = { # 算法配置
    "actor": {
        "number": 1,
        "type": DDPGActor,
        "params": actor_params,
        "networks": ["actor_net"],
        "policies": ['init_policy', 'evaluate_policy', 'collect_policy' ],
    },
    "learner": {
        "number": 1,
        "type": DDPGLearner,
        "params": learner_params,
        "networks": [
            "actor_net",
            "target_actor_net",
            "critic_net",
            "target_critic_net",
        ],
    },
    "policy_and_network": {"type": DDPGPolicy, "params": policy_params},
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
                      'data_shape': [(144,), (1,), (1,), (144,), (1,)],
                      'data_type': [ms.float32, ms.int32, ms.float32, ms.float32, ms.bool_],
                      'sample_size': 32},
}

summary_config = { # 摘要配置
    "mindinsight_on": False,
    "base_dir": "./summary",
    "collect_interval": 2,
}
