# Copyright 2023 Huawei Technologies Co., Ltd
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
"""double_dqn dqn"""

from mindspore import ops
import mindspore
import mindspore as ms

# from mindspore_rl.algorithm.dqn import DQNLearner
from .dqn import DQNLearner
from mindspore_rl.agent.learner import Learner
import mindspore.nn as nn
from mindspore import dtype as mstype

from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter, ParameterTuple

_update_opt = C.MultitypeFuncGraph("update_opt")

@_update_opt.register("Tensor", "Tensor")
def _parameter_update(policy_param, target_param):
    assign = P.Assign()
    new_param = 1 * policy_param
    output = assign(target_param, new_param)
    return output


# class DoubleDQNLearner(DQNLearner):
#     """Double DQN Learner"""

#     class PolicyNetWithLossCell(nn.Cell):
#         """DQN policy network with loss cell"""

#         def __init__(self, backbone, loss_fn):
#             super(DoubleDQNLearner.PolicyNetWithLossCell,
#                   self).__init__(auto_prefix=False)
#             self._backbone = backbone
#             self._loss_fn = loss_fn
#             self.gather = P.GatherD()

#         def construct(self, x, a0, label):
#             """constructor for Loss Cell"""
#             out = self._backbone(x)
#             out = self.gather(out, 1, a0)
#             loss = self._loss_fn(out, label)
#             return loss

#     def __init__(self, params):
#         super().__init__()
#         self.policy_network = params['policy_network']
#         self.target_network = params['target_network']
#         self.policy_param = ParameterTuple(
#             self.policy_network.get_parameters())
#         self.target_param = ParameterTuple(
#             self.target_network.get_parameters())
        
#         optimizer = nn.RMSProp(
#             self.policy_network.trainable_params(),
#             learning_rate=params['lr'])
#         loss_fn = nn.MSELoss()
#         loss_q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
#         self.policy_network_train = nn.TrainOneStepCell(loss_q_net, optimizer)
#         self.policy_network_train.set_train(mode=True)

#         self.gamma = Tensor(params['gamma'], ms.float32)
#         self.expand_dims = P.ExpandDims()
#         self.reshape = P.Reshape()
#         self.hyper_map = C.HyperMap()
#         self.ones_like = P.OnesLike()
#         self.select = P.Select()
#         self.argmax = P.Argmax(output_type=mindspore.int32)

#     def learn(self, experience):
#         """Model update"""
#         s0, a0, r1, s1 = experience
#         policy_state_values = self.policy_network(s1)
#         max_policy_action = ops.argmax(policy_state_values, dim=1, keepdim=True)
#         next_state_values = self.target_network(s1)
#         next_state_values = ops.gather_d(
#             next_state_values, 1, max_policy_action
#         ).squeeze(-1)
#         r1 = self.reshape(r1, (-1,))

#         y_true = r1 + self.gamma * next_state_values

#         # Modify last step reward
#         one = self.ones_like(r1)
#         y_true = self.select(r1 == -one, one, y_true)
#         y_true = self.expand_dims(y_true, 1)

#         success = self.policy_network_train(s0, a0, y_true)
#         return success
    
#     def update(self):
#         """Update the network parameters"""
#         assign_result = self.hyper_map(
#             _update_opt,
#             self.policy_param,
#             self.target_param)
#         # self.steps += 1
#         return assign_result

class DoubleDQNLearner(Learner):
    """Double DQN Learner"""

    class PolicyNetWithLossCell(nn.Cell):
        """DQN policy network with loss cell"""

        def __init__(self, backbone, loss_fn):
            super(DoubleDQNLearner.PolicyNetWithLossCell,
                  self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self.gather = P.GatherD()

        def construct(self, x, a0, label):
            """constructor for Loss Cell"""
            out = self._backbone(x)
            out = self.gather(out, 1, a0)
            loss = self._loss_fn(out, label)
            return loss

    def __init__(self, params):
        super().__init__()
        self.policy_network = params['policy_network']
        self.target_network = params['target_network']
        self.policy_param = ParameterTuple(
            self.policy_network.get_parameters())
        self.target_param = ParameterTuple(
            self.target_network.get_parameters())
        
        optimizer = nn.RMSProp(
            self.policy_network.trainable_params(),
            learning_rate=params['lr'])
        loss_fn = nn.MSELoss()
        loss_q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
        self.policy_network_train = nn.TrainOneStepCell(loss_q_net, optimizer)
        self.policy_network_train.set_train(mode=True)

        self.gamma = Tensor(params['gamma'], ms.float32)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.ones_like = P.OnesLike()
        self.select = P.Select()
        self.argmax = P.Argmax(output_type=mindspore.int32)

    def learn(self, experience):
        """Model update"""
        s0, a0, r1, s1 = experience
        policy_state_values = self.policy_network(s1)
        # max_policy_action = self.argmax(policy_state_values, dim=1, keepdim=True)
        max_policy_action = self.argmax(policy_state_values)
        max_policy_action = self.expand_dims(max_policy_action, -1)
        next_state_values = self.target_network(s1)
        next_state_values = ops.gather_d(
            next_state_values, 1, max_policy_action
        ).squeeze(-1)
        r1 = self.reshape(r1, (-1,))

        y_true = r1 + self.gamma * next_state_values

        # Modify last step reward
        one = self.ones_like(r1)
        y_true = self.select(r1 == -one, one, y_true)
        y_true = self.expand_dims(y_true, 1)

        success = self.policy_network_train(s0, a0, y_true)
        return success
    
    def update(self):
        """Update the network parameters"""
        assign_result = self.hyper_map(
            _update_opt,
            self.policy_param,
            self.target_param)
        # self.steps += 1
        return assign_result
