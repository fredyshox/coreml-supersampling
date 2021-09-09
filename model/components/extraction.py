import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from model.components import CONV2D_KERNEL_SIZE

class FeatureExtractionModule(Model):
    def __init__(self, channel_count=4, name=None):
        super().__init__(name=name)

        input_shape = (None, None, channel_count)
        self.conv_0 = Conv2D(32, CONV2D_KERNEL_SIZE, activation="relu", padding="same", input_shape=input_shape)
        self.conv_1 = Conv2D(32, CONV2D_KERNEL_SIZE, activation="relu", padding="same")
        self.conv_2 = Conv2D(8, CONV2D_KERNEL_SIZE, activation="relu", padding="same")

    def call(self, x):
        h = self.conv_0(x)
        h = self.conv_1(h)
        h = self.conv_2(h)
        output = tf.concat([x, h], axis=3)
        return output