import gymnasium as gym
from typing import Union

# Pick either DL-framework.
import torch
#import tensorflow as tf

from rl_generic.room_env import RoomEnv
from rl_generic.models import MyTorchModel  #, MyKerasModel


class DQN:
    def __init__(self):
        self.env: gym.Env = RoomEnv()
        self.model: Union["torch.nn.Module", "tf.keras.Model"] = MyTorchModel()  # MyKerasModel()

        self.last_obs, _ = self.env.reset()

        self.gamma = 0.99
        self.batch_size = 256
        self.epsilon = 1.0
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters())
        #self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def train(self):
        pass

    def get_action(
        self,
        single_observation: Union["torch.Tensor", "tf.Tensor"],
    ):
        pass

    def loss(
        self,
        obs_batch: Union["torch.Tensor", "tf.Tensor"],
        action_batch: Union["torch.Tensor", "tf.Tensor"],
        reward_batch: Union["torch.Tensor", "tf.Tensor"],
        done_batch: Union["torch.Tensor", "tf.Tensor"],
        next_obs_batch: Union["torch.Tensor", "tf.Tensor"],
    ):
        pass


class VPG:
    def __init__(self):
        self.env: gym.Env = RoomEnv()
        self.model: Union["torch.nn.Module", "tf.keras.Model"] = MyTorchModel()  # MyKerasModel()

        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters())
        #self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def train(self):
        pass

    def get_action(
        self,
        single_observation: Union["torch.Tensor", "tf.Tensor"],
    ):
        pass

    def loss(
        self,
        obs_batch: Union["torch.Tensor", "tf.Tensor"],
        action_batch: Union["torch.Tensor", "tf.Tensor"],
        reward_batch: Union["torch.Tensor", "tf.Tensor"],
    ):
        pass


if __name__ == "__main__":
    dqn = DQN()
    for _ in range(10000):
        dqn.train()