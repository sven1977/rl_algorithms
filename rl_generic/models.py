import numpy as np

# Either one should work.
import torch
#import tensorflow as tf


class MyTorchModel(torch.nn.Module):
    # DO NOT CHANGE. MODEL IS ALREADY COMPLETED.
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(64, 64)
        self.outputs = torch.nn.Linear(64, 4)

    def forward(self, inputs):
        one_hot = torch.nn.functional.one_hot(
          inputs, num_classes=64
        )
        layer1 = self.layer(one_hot.float())
        layer1 = torch.nn.functional.relu(layer1)
        return self.outputs(layer1)


# class MyKerasModel(tf.keras.Model):
#    # DO NOT CHANGE. MODEL IS ALREADY COMPLETED.
#    def __init__(self):
#        super().__init__()
#        self.layer = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
#        self.outputs = tf.keras.layers.Dense(4)

#    def __call__(self, inputs):
#        one_hot = tf.one_hot(inputs, depth=64)
#        layer1 = self.layer(one_hot)
#        return self.outputs(layer1)
