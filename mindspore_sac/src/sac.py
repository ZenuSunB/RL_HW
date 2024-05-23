# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""SAC Agent"""
import mindspore
import numpy as np
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore.nn.probability.distribution as msd
from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils import SoftUpdate
from mindspore.ops import stop_gradient

class SACPolicy:
    """
    This is SACPolicy class. You should define your networks (SACActorNet and SACCriticNet here)
    which you prepare to use in the algorithm. Moreover, you should also define you loss function
    (SACLossCell here) which calculates the loss between policy and your ground truth value.
    """

    class SACActorNet(nn.Cell):
        """
        SACActorNet is the actor network of SAC algorithm. It takes a set of state as input
        and outputs miu, sigma of a normal distribution
        """

        def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            compute_type=mindspore.float32,
        ):
            super(SACPolicy.SACActorNet, self).__init__()
            self.common = nn.Dense(input_size, hidden_size, weight_init='XavierUniform')
            self.actor = nn.Dense(hidden_size, output_size, weight_init='XavierUniform')
            self.relu = nn.LeakyReLU()
            self.softmax = P.Softmax()

        def construct(self, obs):
            """calculate miu and sigma"""
            x = self.common(obs)
            x = self.relu(x)
            x = self.actor(x)
            return self.softmax(x)

    class SACCriticNet(nn.Cell):
        """
        SACCriticNet is the critic network of SAC algorithm. It takes a set of states as input
        and outputs the value of input state
        """

        def __init__(
            self,
            obs_size,
            hidden_size,
            output_size,
            compute_type=mindspore.float32,
        ):
            super(SACPolicy.SACCriticNet, self).__init__()
            self.common = nn.Dense(obs_size, hidden_size, weight_init='XavierUniform')
            self.critic = nn.Dense(hidden_size, output_size, weight_init='XavierUniform')
            self.relu = nn.LeakyReLU()
           

        def construct(self, obs):
            """predict value"""
            x = self.common(obs)
            x = self.relu(x)
            return self.critic(x)

    class RandomPolicy(nn.Cell):
        def __init__(self, action_space_dim):
            super(SACPolicy.RandomPolicy, self).__init__()
            self.uniform = P.UniformReal()
            self.shape = (action_space_dim,)

        def construct(self):
            return self.uniform(self.shape) * 2 - 1

    class CollectPolicy(nn.Cell):
        """Collect Policy"""

        def __init__(self, actor_net):
            super(SACPolicy.CollectPolicy, self).__init__()
            self.actor_net = actor_net
            self.c_dist = msd.Categorical(dtype=mindspore.float32, seed=3407)
            self.reshape = P.Reshape()
            self.cast = P.Cast()
        def construct(self, obs):
            action_probs_t = self.actor_net(obs)
            action = self.reshape(self.c_dist.sample(
                (1,), probs=action_probs_t), (1,))
            action=self.cast(action, mindspore.int32)
            return action

    class EvalPolicy(nn.Cell):
        """Eval Policy"""

        def __init__(self, actor_net):
            super(SACPolicy.EvalPolicy, self).__init__()
            self.actor_net = actor_net
            self.reshape = P.Reshape()
            self.argmax = P.Argmax(output_type=mindspore.int32)
            
        def construct(self, obs):
            action_probs_t = self.actor_net(obs)
            action = self.reshape(self.argmax(action_probs_t), (1,))
            return action

    def __init__(self, params):
        compute_type = params.get("compute_type", mindspore.float32)
        self.actor_net = self.SACActorNet(
            input_size=params["state_space_dim"],
            hidden_size=params["hidden_size"],
            output_size=params["action_space_dim"],
            compute_type=compute_type,
        )
        self.critic_net1 = self.SACCriticNet(
            obs_size=params["state_space_dim"],
            hidden_size=params["hidden_size"],
            output_size=params["action_space_dim"],
            compute_type=compute_type,
        )
        self.critic_net2 = self.SACCriticNet(
            obs_size=params["state_space_dim"],
            hidden_size=params["hidden_size"],
            output_size=params["action_space_dim"],
            compute_type=compute_type,
        )
        self.target_critic_net1 = self.SACCriticNet(
            obs_size=params["state_space_dim"],
            hidden_size=params["hidden_size"],
            output_size=params["action_space_dim"],
            compute_type=compute_type,
        )
        self.target_critic_net2 = self.SACCriticNet(
            obs_size=params["state_space_dim"],
            hidden_size=params["hidden_size"],
            output_size=params["action_space_dim"],
            compute_type=compute_type,
        )

        self.init_policy = self.RandomPolicy(params["action_space_dim"])
        self.collect_policy = self.CollectPolicy(self.actor_net)
        self.eval_policy = self.EvalPolicy(self.actor_net)


class SACActor(Actor):
    """
    This is an actor class of SAC algorithm, which is used to interact with environment, and
    generate/insert experience (data)
    """

    def __init__(self, params=None):
        super().__init__()
        self._params_config = params
        self._environment = params["collect_environment"]
        self._eval_env = params["eval_environment"]
        self.init_policy = params["init_policy"]
        self.collect_policy = params["collect_policy"]
        self.eval_policy = params["eval_policy"]
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=0)
        self.argmax = P.Argmax(output_type=mindspore.int32)
    def act(self, phase, params):
        """collect experience and insert to replay buffer (used during training)"""
        if phase == 1:
            action_probs_t = self.expand_dims(self.init_policy(),0)
            action = self.argmax(action_probs_t)
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 2:
            params = self.expand_dims(params, 0)
            action = self.collect_policy(params)
            new_state, reward, done = self._environment.step(action)
            return new_state, action, reward, done
        if phase == 3:
            params = self.expand_dims(params, 0)
            action = self.eval_policy(params)
            new_state, reward, done = self._eval_env.step(action)
            return reward, new_state, done
        self.print("Phase is incorrect")
        return 0

    def get_action(self, phase, params):
        """get action"""
        obs = self.expand_dims(params, 0)
        action = self.eval_policy(obs)
        return self.squeeze(action,-1)


class SACLearner(Learner):
    """This is the learner class of SAC algorithm, which is used to update the policy net"""

    class CriticLossCell(nn.Cell):
        """CriticLossCell"""

        def __init__(
            self,
            gamma,
            log_alpha,
            actor_net,
            target_critic_net1,
            target_critic_net2,
            critic_net1,
            critic_net2,
        ):
            super(SACLearner.CriticLossCell, self).__init__(auto_prefix=True)
            self.gamma = gamma
            self.log_alpha = log_alpha
            self.actor_net = actor_net
            self.target_critic_net1 = target_critic_net1
            self.target_critic_net2 = target_critic_net2
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.min = P.Minimum()
            self.exp = P.Exp()
            self.mse = nn.MSELoss(reduction="none")
            self.gather = P.GatherD()
            self.log = ops.Log()

        def construct(self, next_state, reward, state, action, done):
            """Calculate critic loss"""
            next_action_probs_t = self.actor_net(next_state)
            next_action_log_probs = self.log(next_action_probs_t+1e-8)
            entropy = -(next_action_probs_t * next_action_log_probs).sum(axis=1, keepdims=True)
            target_q_value1 = self.target_critic_net1(next_state)
            target_q_value2 = self.target_critic_net2(next_state)
            target_q_value = (next_action_probs_t*self.min(target_q_value1, target_q_value2)).sum(axis=1, keepdims=True)
            next_value = target_q_value + self.exp(self.log_alpha) * entropy
            td_target = reward.squeeze(axis=-1) + self.gamma * (1 - done.squeeze(axis=-1)) * next_value.squeeze(axis=-1)
            td_target = stop_gradient(td_target)
            pred_td_target1 = self.gather(self.critic_net1(state), 1, action)
            critic_loss1 = self.mse(td_target, pred_td_target1)
            
            pred_td_target2 =  self.gather(self.critic_net2(state), 1, action)
            critic_loss2 = self.mse(td_target, pred_td_target2)
            
            critic_loss = (critic_loss1 + critic_loss2).mean()
            return critic_loss

    class ActorLossCell(nn.Cell):
        """ActorLossCell"""

        def __init__(
            self,
            log_alpha,
            actor_net,
            critic_net1,
            critic_net2,
        ):
            super(SACLearner.ActorLossCell, self).__init__(auto_prefix=False)
            self.log_alpha = log_alpha
            self.actor_net = actor_net
            self.critic_net1 = critic_net1
            self.critic_net2 = critic_net2
            self.min = P.Minimum()
            self.exp = P.Exp()
            self.log = ops.Log()


        def construct(self, state):
            """Calculate actor loss"""
            action_probs_t = self.actor_net(state)
            action_log_probs = self.log(action_probs_t + 1e-8)
            entropy = - (action_probs_t * action_log_probs).sum(axis=1, keepdims=True)
            target_q_value1 = self.critic_net1(state)
            target_q_value2 = self.critic_net2(state)
            target_q_value = (action_probs_t*self.min(target_q_value1, target_q_value2)).sum(axis=1, keepdims=True)
            actor_loss = (-self.exp(self.log_alpha) * entropy - target_q_value).mean()
            return actor_loss

    class AlphaLossCell(nn.Cell):
        """AlphaLossCell"""

        def __init__(self, 
                     log_alpha, 
                     target_entropy, 
                     actor_net):
            super(SACLearner.AlphaLossCell, self).__init__(auto_prefix=False)
            self.log_alpha = log_alpha
            self.target_entropy = target_entropy
            self.actor_net = actor_net
            self.log = ops.Log()    
            self.exp = P.Exp()
        def construct(self, state):
            action_probs_t = self.actor_net(state)
            action_log_probs = self.log(action_probs_t + 1e-8)
            entropy = -(action_probs_t * action_log_probs).sum(axis=1, keepdims=True)
            entropy_target = stop_gradient(entropy-self.target_entropy)
            alpha_loss = (entropy_target * self.exp(self.log_alpha)).mean()
            return alpha_loss

    def __init__(self, params):
        super().__init__()
        self._params_config = params
        gamma = Tensor(self._params_config["gamma"], mindspore.float32)
        actor_net = params["actor_net"]
        critic_net1 = params["critic_net1"]
        critic_net2 = params["critic_net2"]
        target_critic_net1 = params["target_critic_net1"]
        target_critic_net2 = params["target_critic_net2"]

        log_alpha = params["log_alpha"]
        log_alpha = Parameter(
            Tensor(
                [
                    log_alpha,
                ],
                mindspore.float32,
            ),
            name="log_alpha",
            requires_grad=True,
        )

        critic_loss_net = SACLearner.CriticLossCell(
            gamma,
            log_alpha,
            actor_net,
            target_critic_net1,
            target_critic_net2,
            critic_net1,
            critic_net2,
        )
        actor_loss_net = SACLearner.ActorLossCell(
            log_alpha,
            actor_net,
            critic_net1,
            critic_net2,
        )

        critic_trainable_params = (
            critic_net1.trainable_params() + critic_net2.trainable_params()
        )
        critic_optim = nn.Adam(
            critic_trainable_params, learning_rate=params["critic_lr"], eps=1e-5
        )
        actor_optim = nn.Adam(
            actor_net.trainable_params(), learning_rate=params["actor_lr"], eps=1e-5
        )

        self.critic_train = nn.TrainOneStepCell(critic_loss_net, critic_optim)
        self.actor_train = nn.TrainOneStepCell(actor_loss_net, actor_optim)

        self.train_alpha_net = params["train_alpha_net"]
        if self.train_alpha_net:
            alpha_loss_net = SACLearner.AlphaLossCell(
                log_alpha,
                params["target_entropy"],
                actor_net,
            )
            alpha_optim = nn.Adam([log_alpha], learning_rate=params["alpha_lr"], eps=1e-5)
            self.alpha_train = nn.TrainOneStepCell(alpha_loss_net, alpha_optim)

        factor, interval = params["update_factor"], params["update_interval"]
        params = critic_net1.trainable_params() + critic_net2.trainable_params()
        target_params = (
            target_critic_net1.trainable_params()
            + target_critic_net2.trainable_params()
        )
        self.soft_updater = SoftUpdate(factor, interval, params, target_params)

    def learn(self, experience):
        """learn"""
        state, action, reward, next_state, done = experience

        critic_loss = self.critic_train(next_state, reward, state, action, done)
        actor_loss = self.actor_train(state)

        alpha_loss = 0.0
        if self.train_alpha_net:
            alpha_loss = self.alpha_train(state)

        self.soft_updater()
        loss = critic_loss + actor_loss + alpha_loss
        return loss
