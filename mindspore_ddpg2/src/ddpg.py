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
"""DDPG"""
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

    
class DDPGSoftUpdate(SoftUpdate):  # 实现了软更新策略，用于在训练过程中平滑地更新目标网络的参数
    def __init__(self, factor, update_interval, behavior_params, target_params):
        super().__init__(factor, update_interval, behavior_params, target_params)
        self.steps = Parameter(
            initializer(1, [1], mindspore.int32),
            name="private_steps",
            requires_grad=False,
        )

class GaussianNoise(nn.Cell): #实现了高斯噪声，用于为动作添加随机扰动，从而增加探索性
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

class HuberLoss(nn.Cell): #实现了Huber损失函数，用于训练过程中的误差计算，Huber损失在误差较小时与L2损失相同，在误差较大时与L1损失相同，从而对离群点更具鲁棒性
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

class DDPGPolicy: # 定义了算法中的Actor和Critic网络。Actor网络用于选择动作，Critic网络用于评估动作的价值。每个网络都有对应的目标网络，用于提高训练的稳定性

    class DDPGActorNet(nn.Cell): #actor网络，是一个三层的全连接神经网络。网络结构包括两个隐藏层和一个输出层

        def __init__(
            self,
            input_size,
            hidden_size1,
            hidden_size2,
            output_size,
            compute_type=mindspore.float32,
            name=None,
        ):
            super(DDPGPolicy.DDPGActorNet, self).__init__()

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
            self.soft_max = ops.Softmax()
            self.relu = P.ReLU()
            self.gumbel_softmax = ops.gumbel_softmax
            

        def construct(self, x):
            x = self.relu(self.dense1(x))
            # x = self.relu(self.dense2(x))
            # x = self.tanh(self.dense3(x))
            # x = self.soft_max(self.dense3(x))
            x = self.gumbel_softmax(self.dense3(x),0.01,True)
            return x

    class DDPGCriticNet(nn.Cell): #critic网络，也是一个三层的全连接神经网络。输入是观察值和动作的拼接，输出是Q值。网络结构包括两个隐藏层和一个输出层

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
            super(DDPGPolicy.DDPGCriticNet, self).__init__()

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
                hidden_size2 + action_size, output_size, weight_init=last_weight_init
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
            # q = self.relu(self.dense2(q))
            q = self.dense3(q)

            return q

        
    def __init__(self, params):
        self.actor_net = self.DDPGActorNet( # actor网络
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
            name="actor_net.",
        )
        self.target_actor_net = self.DDPGActorNet( # actor的target网络
            params["state_space_dim"],
            params["hidden_size1"],
            params["hidden_size2"],
            params["action_space_dim"],
            params["compute_type"],
            name="target_actor_net.",
        )
        self.critic_net = self.DDPGCriticNet( # critic网络
            params["state_space_dim"],
            1,
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="critic_net",
        )
        self.target_critic_net = self.DDPGCriticNet( # critic的目标网络
            params["state_space_dim"],
            1,
            params["hidden_size1"],
            params["hidden_size2"],
            1,
            params["compute_type"],
            name="target_critic_net",
        )
        self.init_policy = RandomPolicy(params['action_space_dim']) #初始化阶段的策略
        self.collect_policy = self.actor_net # 数据收集策略：使用actor网络
        self.evaluate_policy = GreedyPolicy(self.actor_net) # 评估策略在actor网络的基础上采用贪婪
        
class DDPGActor(Actor): #继承自MindSpore RL中的Actor类，并实现了act方法。这个方法根据不同的阶段（phase）来选择不同的策略与环境进行交互

    def __init__(self, params):
        super(DDPGActor,self).__init__()
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
            # Fill the replay buffer 填充经验回放缓冲区。此时使用初始化策略来选择动作，执行环境步骤，并将结果存储在经验回放缓冲区中
            action = self.init_policy()
            new_state, reward, done = self.env.step(action)
            action = self.reshape(action, (1,))
            action = self.cast(action, mindspore.int32)
            my_reward = reward
            return done, reward, new_state, action, my_reward
        if phase == 2:
          # Sample action to act in env 在环境中采样动作。此时使用数据收集策略（即actor网络）来选择动作，并添加探索噪声。动作被限制在环境的动作空间范围内，然后执行环境步骤
            ts0 = self.expand_dims(params, 0)
            action_probs = self.collect_policy(ts0)
            action = self.argmax(action_probs)
            action += self.noise(action)
            action = C.clip_by_value(action, self.clip_value_min, self.clip_value_max)
            action = self.cast(action, mindspore.int32)
            new_state, reward, done = self.env.step(action)
            my_reward = reward
            return done, reward, new_state, action, my_reward
        if phase == 3:
            # Evaluate the trained policy 评估训练策略。此时使用评估策略（即贪婪策略）来选择动作，并在评估环境中执行步骤
            ts0 = self.expand_dims(params, 0)
            action = self.evaluate_policy(ts0)
            new_state, reward, done = self.eval_env.step(
                self.cast(action, mindspore.int32))
            my_reward = reward
            return done, reward, new_state, action, my_reward
        
        self.print("Phase is incorrect")
        return 0

class DDPGLearner(Learner):

    class CriticLossCell(nn.Cell): # 计算critic网络的损失

        def __init__(
            self,
            gamma,
            noise_stddev,
            noise_clip,
            action_boundary,
            target_actor_net,
            target_critic_net,
            critic_net
        ):
            super(DDPGLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.target_actor_net = target_actor_net
            self.target_critic_net = target_critic_net
            self.critic_net = critic_net
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
            self.cast = P.Cast()
            self.soft_max = ops.Softmax()
            self.expand_dims = P.ExpandDims()
        def construct(self, obs, action, rewards, next_obs, done):
            """calculate the critic loss"""
            target_action_probs = self.target_actor_net(next_obs)
            target_action = self.argmax(target_action_probs)
            # noisy_target_action = target_action + self.noises(target_action)
            # noisy_target_action = C.clip_by_value(
            #     noisy_target_action, self.low, self.high
            # )
            # noisy_target_action = self.cast(noisy_target_action,mindspore.int32)
            # noisy_target_action = self.expand_dims(noisy_target_action, -1)
            target_action = self.expand_dims(target_action, -1)
            target_q_values = self.target_critic_net(next_obs, target_action)
            
            td_targets = rewards + self.gamma * (1.0 - done) * target_q_values

            # predicted values
            pred_q = self.critic_net(obs, action)
            critic_loss = self._loss(pred_q, td_targets) 
            return critic_loss

    class ActorLossCell(nn.Cell): # 计算actor网络的损失

        def __init__(self, actor_net, critic_net):
            super(DDPGLearner.ActorLossCell, self).__init__(auto_prefix=True)
            self.actor_net = actor_net
            self.critic_net = critic_net
            self.reduce_mean = P.ReduceMean()
            self.argmax = P.Argmax(output_type=mindspore.int32)
            self.reshape = P.Reshape()
            self.one_hot = ops.OneHot()
            self.soft_max = ops.Softmax()
            self.expand_dims = P.ExpandDims()
            self.indices = Parameter(initializer(self.expand_dims(ops.arange(0,40),0),[1,40], mindspore.int32), "indices",False,)
        def construct(self, obs):
            """calculate the actor loss"""
            action_probs = self.actor_net(obs)
            action_soft_onehot_M_indices = action_probs * self.indices
            action = action_soft_onehot_M_indices.sum(axis=-1,keepdims=True) 
            q_values = self.critic_net(obs, action)
            q_values = - q_values
            actor_loss = self.reduce_mean(q_values)
            return actor_loss

    def __init__(self, params):
        super().__init__()
        gamma = params["gamma"] #奖励衰减系数
        noise_stddev = params["target_action_noise_stddev"] #目标动作噪声的标准差
        noise_clip = params["target_action_noise_clip"] #目标动作噪声的剪切
        self.critic_net = params["critic_net"]
        self.actor_net = params["actor_net"]
        self.actor_update_interval = params["actor_update_interval"] # actor网络更新间隔
        self.action_boundary = params["action_boundary"] #动作边界

        # util
        self.mod = P.Mod()
        self.equal = P.Equal()

        # optimizer network 优化器
        critic_optimizer = nn.Adam(
            self.critic_net.trainable_params(),
            learning_rate=params["critic_lr"], eps=1e-5
        )
        actor_optimizer = nn.Adam(
            self.actor_net.trainable_params(), learning_rate=params["actor_lr"], eps=1e-5
        )

        # target networks and their initializations 目标网络与参数初始化
        self.target_actor_net = params["target_actor_net"]
        self.target_critic_net = params["target_critic_net"]
        behave_params = (
            self.actor_net.trainable_params()
            + self.critic_net.trainable_params()
        )
        target_params = (
            self.target_actor_net.trainable_params()
            + self.target_critic_net.trainable_params()
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
            self.target_critic_net,
            self.critic_net,
        )
        self.actor_loss_cell = self.ActorLossCell(self.actor_net, self.critic_net)

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
        self.soft_updater = DDPGSoftUpdate( # 目标网络更新器，采用软更新
            factor, interval, behave_params, target_params
        )

    def learn(self, experience): # 执行算法的训练过程
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
