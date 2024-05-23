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
"""DDPG Trainer"""
import mindspore
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Parameter, Tensor
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer
from .ddpg_summary import RecordQueue


class DDPGTrainer(Trainer):

    def __init__(self, msrl, params=None):
        nn.Cell.__init__(self, auto_prefix=False)
        self.zero = Tensor(0, mindspore.float32)
        self.zero_value = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.num_eval_episode = params["num_eval_episode"]
        self.steps = Parameter(Tensor([1], mindspore.int32))
        self.true = Tensor(True, mindspore.bool_)
        self.false = Tensor([False], mindspore.bool_)
        self.init_collect_size = Tensor(params["init_collect_size"], mindspore.float32)
        self.fill_value = Tensor(params['buffer_num_before_learning_begin'], mindspore.float32)
        self.inited = Parameter(Tensor(False, mindspore.bool_), name="init_flag")
        if "eval_episodes" in params:
            self.eval_episodes = params["eval_episodes"]
        self.update_period = Tensor(params['update_target_iter'], mindspore.float32)
        super(DDPGTrainer, self).__init__(msrl)
            

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"actor_net": self.msrl.actors.actor_net}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero_value
        while self.less(i, self.init_collect_size):
            done, _, new_state, action, my_reward = self.msrl.agent_act(
                trainer.COLLECT, state)
            if not done:
                self.msrl.replay_buffer_insert(
                    [state, action, my_reward, new_state, done])
                state = new_state
            else:
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
        state = self.msrl.collect_environment.reset()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero
        while not done:
            done, r, new_state, action, my_reward = self.msrl.agent_act(
                trainer.COLLECT, state)
            if done:
                break
            self.msrl.replay_buffer_insert([state, action, my_reward, new_state, done])
            state = new_state
            r = self.squeeze(r)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += r
            self.steps += 1
            steps += 1
        return loss, total_reward, steps

    @ms_function
    def evaluate(self):
        """evaluate function"""
        total_reward = self.zero_value
        eval_iter = self.zero_value
        while self.less(eval_iter, self.num_eval_episode):
            episode_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                done, r, state, _, _ = self.msrl.agent_act(trainer.EVAL, state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_eval_episode
        return avg_reward

    def load_and_eval(self, ckpt_path=None):
        """
        The interface of the eval function for offline. A checkpoint must be provided.

        Args:
            ckpt_path (string): The checkpoint file to restore net.
        """
        if ckpt_path is None:
            raise RuntimeError("Please provide a ckpt_path.")
        self._init_or_restore(ckpt_path)
        if self.eval_episodes <= 0:
            raise ValueError(
                "In order to get average rewards,\
                evaluate episodes should be larger than 0, but got {}".format(
                    self.eval_episodes
                )
            )
        rewards = RecordQueue()
        for _ in range(self.eval_episodes):
            reward = self.evaluate()
            rewards.add(reward)
            print(reward)
        avg_reward = rewards.mean().asnumpy()
        print("-----------------------------------------")
        print(
            f"Average evaluate result is {avg_reward:.3f}, checkpoint file in {ckpt_path}"
        )
        print("-----------------------------------------")