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
"""TD3"""
import numpy as np
import mindspore
import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import Parameter, Tensor, nn
from mindspore.common.initializer import Uniform, VarianceScaling, initializer
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.ops as ops

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import SoftUpdate
from mindspore_rl.policy import RandomPolicy
from mindspore_rl.policy import GreedyPolicy


seed = 3407
np.random.seed(seed)
class TD3SoftUpdate(SoftUpdate):
    def __init__(self, factor, update_interval, behavior_params, target_params):
        super().__init__(factor, update_interval, behavior_params, target_params)
        self.steps = Parameter(
            initializer(1, [1], mindspore.int32),
            name="private_steps",
            requires_grad=False,
        )

class GaussianNoise(nn.Cell):
    """Noise class applied Normal distribution"""

    def __init__(self, mean, stddev, clip=None):
        super().__init__()
        self.clip = clip
        if self.clip is not None:
            self.high_clip = Tensor(np.abs(self.clip))
            self.low_clip = Tensor(-np.abs(self.clip))
        self.normal = msd.Normal(mean, stddev)

    def construct(self, actions):
        noises = self.normal.sample(actions.shape)
        if self.clip is not None:
            noises = C.clip_by_value(noises, self.low_clip, self.high_clip)
        return noises

class HuberLoss(nn.Cell):
    """Huber Loss"""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = Tensor(delta, mindspore.float32)
        self.abs = P.Abs()
        self.square = P.Square()
        self.select = P.Select()
        self.reduce_mean = P.ReduceMean()

    def construct(self, predict, label):
        abs_error = self.abs(predict - label)
        cond = abs_error <= self.delta
        loss = self.select(
            cond,
            0.5 * self.square(abs_error),
            self.delta * abs_error - 0.5 * self.square(self.delta),
        )
        return self.reduce_mean(loss)

class TD3Policy:
    """TD3 Policy"""

    class TD3ActorNet(nn.Cell):
        """TD3 Actor Network"""

        def __init__(
            self,
            input_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
            name=None,
        ):
            super(TD3Policy.TD3ActorNet, self).__init__()

            weight_init = VarianceScaling(
                scale=1.0 / 3, mode="fan_in", distribution="uniform"
            )
            self.dense1 = nn.Dense(
                input_size, hidden_size1, weight_init=weight_init
            ).to_float(compute_type)
            self.dense2 = nn.Dense(
                hidden_size1, hidden_size2, weight_init=weight_init
            ).to_float(compute_type)
            last_weight_init = Uniform(scale=0.003)
            self.dense3 = nn.Dense(
                hidden_size2, output_size, weight_init=last_weight_init
            ).to_float(compute_type)

            if name is not None:
                self._update_local_parameters_name(prefix=name)
            self.tanh = P.Tanh()
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.tanh(self.dense3(x))
            return x

    class TD3CriticNet(nn.Cell):
        """TD3 Critic Network"""

        def __init__(
            self,
            obs_size,
            action_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
            name=None,
        ):
            super(TD3Policy.TD3CriticNet, self).__init__()

            weight_init = VarianceScaling(
                scale=1.0 / 3, mode="fan_in", distribution="uniform"
            )
            self.dense1 = nn.Dense(
                obs_size, hidden_size1, weight_init=weight_init
            ).to_float(compute_type)
            self.dense2 = nn.Dense(
                hidden_size1 + action_size, hidden_size2, weight_init=weight_init
            ).to_float(compute_type)
            last_weight_init = Uniform(scale=0.003)
            self.dense3 = nn.Dense(
                hidden_size2, output_size, weight_init=last_weight_init
            ).to_float(compute_type)
 
            if name is not None:
                self._update_local_parameters_name(prefix=name)

            # utils
            self.concat = P.Concat(axis=-1)
            self.relu = P.ReLU()
            self.cast = P.Cast()

        def construct(self, observation, action):
            q = self.relu(self.dense1(observation))
            action = self.cast(action, q.dtype)
            q = self.concat((q, action))
            q = self.relu(self.dense2(q))
            q = self.dense3(q)

            return q

        
    def __init__(self, params):
        self.actor_net = self.TD3ActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
            name="actor_net.",
        )
        self.target_actor_net = self.TD3ActorNet(
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
            name="target_actor_net.",
        )
        self.critic_net_1 = self.TD3CriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="critic_net_1.",
        )
        self.critic_net_2 = self.TD3CriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="critic_net_2.",
        )
        self.target_critic_net_1 = self.TD3CriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="target_critic_net_1.",
        )
        self.target_critic_net_2 = self.TD3CriticNet(
            params["state_space_dim"],
            params["action_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="target_critic_net_2.",
        )
        self.init_policy = RandomPolicy(params['action_space_dim'])
        self.collect_policy = self.actor_net
        self.evaluate_policy = GreedyPolicy(self.actor_net)
        
class TD3Actor(Actor):
    """TD3 Actor"""

    def __init__(self, params):
        super(TD3Actor,self).__init__()
        self._params_config = params
        self.init_policy = params["init_policy"]
        self.collect_policy = params['collect_policy']
        self.evaluate_policy = params['evaluate_policy']
        self.env = params["collect_environment"]
        self.eval_env = params['eval_environment']
        self.replay_buffer = params['replay_buffer']
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        
        self.actor_net = params["actor_net"]
        self.squeeze = P.Squeeze()
        low, high = self.env.action_space.boundary
        self.clip_value_min = Tensor(low)
        self.clip_value_max = Tensor(high)
        self.noise = GaussianNoise(0.0, params["actor_explore_noise"])
        self.cast = P.Cast()
        self.print = P.Print()
        self.argmax = P.Argmax(output_type=mindspore.int32)
        self.one_hot = ops.OneHot()
        
    def act(self, phase, params):
        if phase == 1:
            # Fill the replay buffer
            action = self.init_policy()
            new_state, reward, done = self.env.step(action)
            action = self.reshape(action, (1,))
            action = self.cast(action, mindspore.int32)
            action_probs =  self.one_hot(action, 40,Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
            action_probs =  self.squeeze(action_probs)
            my_reward = reward
            return done, reward, new_state, action_probs, my_reward
        if phase == 2:
            # Sample action to act in env
            ts0 = self.expand_dims(params, 0)
            action_probs = self.collect_policy(ts0)
            action_probs = self.squeeze(action_probs)
            action_probs += self.noise(action_probs)
            action_probs = C.clip_by_value(
                action_probs, self.clip_value_min, self.clip_value_max
            )
            action_probs = self.expand_dims(action_probs, 0)
            action = self.argmax(action_probs)
            new_state, reward, done = self.env.step(
                self.cast(action, mindspore.int32))
            action_probs =  self.squeeze(action_probs)
            my_reward = reward
            return done, reward, new_state, action_probs, my_reward
        if phase == 3:
            # Evaluate the trained policy
            ts0 = self.expand_dims(params, 0)
            action = self.evaluate_policy(ts0)
            new_state, reward, done = self.eval_env.step(
                self.cast(action, mindspore.int32))
            my_reward = reward
            return done, reward, new_state, action, my_reward
        
        self.print("Phase is incorrect")
        return 0

class TD3Learner(Learner):
    """TD3 Learner"""

    class CriticLossCell(nn.Cell):
        """Compute the loss of critic network in TD3 algorithm"""

        def __init__(
            self,
            gamma,
            noise_stddev,
            noise_clip,
            action_boundary,
            target_actor_net,
            target_critic_net_1,
            critic_net_1,
            target_critic_net_2,
            critic_net_2,
        ):
            super(TD3Learner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.target_actor_net = target_actor_net
            self.target_critic_net_1 = target_critic_net_1
            self.critic_net_1 = critic_net_1
            self.target_critic_net_2 = target_critic_net_2
            self.critic_net_2 = critic_net_2
            self._loss = HuberLoss()

            # utils
            self.min = P.Minimum()
            self.reduce_mean = P.ReduceMean()
            self.abs = P.Abs()
            self.ones = P.Ones()
            self.noises = GaussianNoise(0.0, noise_stddev, noise_clip)
            low, high = action_boundary
            self.low = Tensor(low)
            self.high = Tensor(high)
            self.argmax = P.Argmax(output_type=mindspore.int32)
            self.squeeze = P.Squeeze()
            self.reshape = P.Reshape()
            self.one_hot = ops.OneHot()
            
        def construct(self, obs, action_probs, rewards, next_obs, done):
            """calculate the critic loss"""
            target_action_probs = self.target_actor_net(next_obs)
            noisy_target_action_probs = target_action_probs + self.noises(target_action_probs)
            noisy_target_action_probs = C.clip_by_value(
                noisy_target_action_probs, self.low, self.high
            )
            # noisy_target_action = self.argmax(noisy_target_action_probs)
            # noisy_target_action_one_hot = self.one_hot(noisy_target_action,40,Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
            # noisy_target_action_batch_vector = noisy_target_action_one_hot
            noisy_target_action_batch_vector = noisy_target_action_probs
            target_q1_values = self.target_critic_net_1(next_obs, noisy_target_action_batch_vector)
            target_q2_values = self.target_critic_net_2(next_obs, noisy_target_action_batch_vector)
            target_q_values = self.min(target_q1_values, target_q2_values)

            td_targets = rewards + self.gamma * (1.0 - done) * target_q_values
            action_probs = self.squeeze(action_probs)
            # predicted values
            pred_q1 = self.critic_net_1(obs, action_probs)
            pred_q2 = self.critic_net_2(obs, action_probs)
            critic_loss = self._loss(pred_q1, td_targets) + self._loss(pred_q2, td_targets)
            return critic_loss

    class ActorLossCell(nn.Cell):
        """ActorLossCell calculates the loss of TD3 algorithm"""

        def __init__(self, actor_net, critic_net):
            super(TD3Learner.ActorLossCell, self).__init__(auto_prefix=True)
            self.actor_net = actor_net
            self.critic_net = critic_net
            self.reduce_mean = P.ReduceMean()
            self.argmax = P.Argmax(output_type=mindspore.int32)
            self.reshape = P.Reshape()
            self.one_hot = ops.OneHot()

        def construct(self, obs):
            """calculate the actor loss"""
            action_probs = self.actor_net(obs)
            # action = self.argmax(action_probs)
            # action_one_hot = self.one_hot(action,40,Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
            # action_batch_vector = action_one_hot
            action_batch_vector = action_probs
            q_values = self.critic_net(obs, action_batch_vector)
            q_values = -q_values
            actor_loss = self.reduce_mean(q_values)
            return actor_loss

    def __init__(self, params):
        super().__init__()
        gamma = params["gamma"]
        noise_stddev = params["target_action_noise_stddev"]
        noise_clip = params["target_action_noise_clip"]
        self.critic_net_1 = params["critic_net_1"]
        self.critic_net_2 = params["critic_net_2"]
        self.actor_net = params["actor_net"]
        self.actor_update_interval = params["actor_update_interval"]
        self.action_boundary = params["action_boundary"]

        # util
        self.mod = P.Mod()
        self.equal = P.Equal()

        # optimizer network
        critic_optimizer = nn.Adam(
            self.critic_net_1.trainable_params() + self.critic_net_2.trainable_params(),
            learning_rate=params["critic_lr"], eps=1e-5
        )
        actor_optimizer = nn.Adam(
            self.actor_net.trainable_params(), learning_rate=params["actor_lr"], eps=1e-5
        )

        # target networks and their initializations
        self.target_actor_net = params["target_actor_net"]
        self.target_critic_net_1 = params["target_critic_net_1"]
        self.target_critic_net_2 = params["target_critic_net_2"]
        behave_params = (
            self.actor_net.trainable_params()
            + self.critic_net_1.trainable_params()
            + self.critic_net_2.trainable_params()
        )
        target_params = (
            self.target_actor_net.trainable_params()
            + self.target_critic_net_1.trainable_params()
            + self.target_critic_net_2.trainable_params()
        )

        self.trainable_params_init = SoftUpdate(
            factor=1.0,
            update_interval=1.0,
            behavior_params=behave_params,
            target_params=target_params,
        )
        self.trainable_params_init()

        # loss cell
        self.critic_loss_cell = self.CriticLossCell(
            gamma,
            noise_stddev,
            noise_clip,
            self.action_boundary,
            self.target_actor_net,
            self.target_critic_net_1,
            self.critic_net_1,
            self.target_critic_net_2,
            self.critic_net_2,
        )
        self.actor_loss_cell = self.ActorLossCell(self.actor_net, self.critic_net_1)

        self.critic_train = nn.TrainOneStepCell(self.critic_loss_cell, critic_optimizer)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_cell, actor_optimizer)
        self.critic_train.set_train(mode=True)
        self.actor_train.set_train(mode=True)

        self.zero = Tensor(0, mindspore.float32)
        self.step = Parameter(
            initializer(0, [1], mindspore.int32),
            name="global_step",
            requires_grad=False,
        )
        self.plus = P.AssignAdd()

        factor, interval = (
            params["target_update_factor"],
            params["target_update_interval"],
        )
        self.soft_updater = TD3SoftUpdate(
            factor, interval, behave_params, target_params
        )

    def learn(self, experience):
        """TD3 learners"""
        self.plus(self.step, 1)
        obs, action, rewards, next_obs, done = experience
        critic_loss = self.critic_train(obs, action, rewards, next_obs, done)

        actor_update_condition = self.mod(self.step, self.actor_update_interval)
        if self.equal(actor_update_condition, self.zero):
            actor_loss = self.actor_train(obs)
        else:
            actor_loss = self.actor_loss_cell(obs)
            
        self.soft_updater()
        total_loss = critic_loss + actor_loss
        return total_loss