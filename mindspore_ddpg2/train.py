# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
DDPG training example.
"""

#pylint: disable=C0413
import argparse
from src import config
from src.ddpg_trainer import DDPGTrainer
from mindspore import context
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DDPG') #创建一个命令行参数解析器，用于解析命令行参数
parser.add_argument('--episode', type=int, default=2000, help='total episode numbers.') #添加一个命令行参数，用于设置总的训练回合数
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'], #添加一个命令行参数，用于选择运行示例的目标设备
                    help='Choose a device to run the ddpg example(Default: Auto).')
options, _ = parser.parse_known_args() #解析命令行参数，并存储在options变量中

def train(episode=options.episode):
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)
    context.set_context(mode=context.GRAPH_MODE) #设置MindSpore的运行模式为图模式
    ddpg_session = Session(config.algorithm_config) # 创建一个MindSpore RL的会话实例，用于执行训练过程
    loss_cb = LossCallback() #创建一个损失回调实例，用于记录和打印损失值
    ckpt_cb = CheckpointCallback(50, config.trainer_params['ckpt_path']) #创建一个检查点回调实例，用于在每50个回合保存一次检查点
    eval_cb = EvaluateCallback(10) #创建一个评估回调实例，用于在每10个回合进行一次评估
    cbs = [loss_cb, ckpt_cb, eval_cb] #创建一个回调列表，包含损失回调、检查点回调和评估回调
    ddpg_session.run(class_type=DDPGTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)


if __name__ == "__main__":
    train()
