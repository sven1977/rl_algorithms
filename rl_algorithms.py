# Given a simple grid-worl/robot navigation problem (see map below):
# ----------   R=robot, O=obstacle, G=goal
# |R    OO |
# |     OO |
# |O       |
# |        |
# |    O   |
# |  OOO   |
# |        |
# |  O    G|
# ----------
# .. implement a very simple Q-learning algorithm or - alternatively - a very simple
# policy gradient algorithm and try to run it and have it learn how to act
# optimally in the environment. In case you chose the policy gradient solution,
# you won't need to implement an extra value function (also no value function loss
# term, just use the raw environment returns).

# Also:
# - Chose a deep learning framework (tf or torch).
# - Use the already provided environment class `RoomEnv` and one of the given
#   model classes (`MyKerasModel`, `MyTorchModel`). See the already provided imports
#   below.

import gym
from typing import Union

# Pick either DL-framework.
import torch
import tensorflow as tf

from rl_generic.room_env import RoomEnv
from rl_generic.models import MyKerasModel


class DQN:
    def __init__(self):
        self.env: gym.Env = RoomEnv()
        self.model: tf.keras.Model = MyKerasModel()  # MyTorchModel()

        self.last_obs: Union[tf.Tensor, torch.Tensor] = self.env.reset()

        self.gamma = 0.99
        self.batch_size = 256
        self.epsilon = 1.0
        self.learning_rate = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        #self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters())

    def train(self):
        pass

    def get_action(self,
                   single_observation: Union[tf.Tensor, torch.Tensor]):
        pass

    def loss(self,
             obs_batch: Union[tf.Tensor, torch.Tensor],
             action_batch: Union[tf.Tensor, torch.Tensor],
             reward_batch: Union[tf.Tensor, torch.Tensor],
             done_batch: Union[tf.Tensor, torch.Tensor],
             next_obs_batch: Union[tf.Tensor, torch.Tensor]
             ):
        pass


class VPG:
    def __init__(self):
        self.env: gym.Env = RoomEnv()
        self.model: Union[tf.keras.Model, torch.nn.Module] = MyKerasModel()  # MyTorchModel()

        self.last_obs: Union[tf.Tensor, torch.Tensor] = self.env.reset()

        self.gamma = 0.99
        self.learning_rate = 2e-5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        #self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters())

    def train(self):
        pass

    def get_action(self,
                   single_observation: Union[tf.Tensor, torch.Tensor]):
        pass

    def loss(self,
             obs_batch: Union[tf.Tensor, torch.Tensor],
             action_batch: Union[tf.Tensor, torch.Tensor],
             reward_batch: Union[tf.Tensor, torch.Tensor]
             ):
        pass


if __name__ == "__main__":
    dqn = DQN()
    for _ in range(10000):
        dqn.train()
