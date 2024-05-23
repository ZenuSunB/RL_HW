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
"""SAC Trainer"""
import mindspore
from mindspore import Parameter, Tensor

from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer

from mindspore.common.api import ms_function


# pylint: disable=W0212
class SACTrainer(Trainer):
    """This is the trainer class of SAC algorithm. It arranges the SAC algorithm"""

    def __init__(self, msrl, params=None):
        nn.Cell.__init__(self, auto_prefix=False)
        self.inited = Parameter(Tensor([False], mindspore.bool_), name="init_flag")
        self.zero = Tensor([0], mindspore.float32)
        self.zero_value = Tensor(0, mindspore.float32)
        self.false = Tensor([False], mindspore.bool_)
        self.true = Tensor([True], mindspore.bool_)
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.duration = params["duration"]
        self.num_eval_episode = params["num_eval_episode"]
        self.squeeze = P.Squeeze()
        super(SACTrainer, self).__init__(msrl)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.actors.collect_policy.actor_net}
        return trainable_variables
    @ms_function
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = Tensor([0], mindspore.int32)
        while self.less(i, Tensor([32], mindspore.int32)):
            new_state, action, reward, done = self.msrl.agent_act(trainer.INIT, state)
            self.msrl.replay_buffer_insert([state, action, reward, new_state, done])
            state = new_state
            if done:
                state = self.msrl.collect_environment.reset()
                done = self.false
            i += 1
        return done

    @ms_function
    def train_one_episode(self):
        """the algorithm in one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true
        done = self.false
        loss = self.zero
        step = self.zero
        total_reward = self.zero
        state = self.msrl.collect_environment.reset()
        while not done:
            new_state, action, reward, done = self.msrl.agent_act(
                trainer.COLLECT, state
            )
            self.msrl.replay_buffer_insert([state, action, reward, new_state, done])
            state = new_state
            total_reward += reward
            batched_transition = self.msrl.replay_buffer_sample()
            loss += self.msrl.agent_learn(batched_transition)
            step += 1
        return loss / step, total_reward, step

    @ms_function
    def evaluate(self):
        """evaluate function"""
        total_eval_reward = self.zero_value
        num_eval = self.zero
        while num_eval < self.num_eval_episode:
            eval_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                reward, state, done = self.msrl.agent_act(trainer.EVAL, state)
                reward = self.squeeze(reward)
                eval_reward += reward
            total_eval_reward += eval_reward
            num_eval += 1
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return  avg_eval_reward
