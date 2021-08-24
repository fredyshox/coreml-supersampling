import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras.models import Sequential
from model import KERNEL_SIZE

class ReconstructionModule4X(tf.Module):
    def __init__(self, frame_count, channels_per_frame=12, name=None):
        super().__init__(name=name)
        
        input_shape = (None, None, frame_count * channels_per_frame)
        self.frame_count = frame_count
        self.channels_per_frame = channels_per_frame
        self.encoder_0 = Sequential([
            Conv2D(64, KERNEL_SIZE, activation="relu", padding="same", input_shape=input_shape),
            Conv2D(32, KERNEL_SIZE, activation="relu", padding="same")
        ])
        self.encoder_0_pooling = MaxPool2D()
        self.encoder_1 = Sequential([
            Conv2D(64, KERNEL_SIZE, activation="relu", padding="same"),
            Conv2D(64, KERNEL_SIZE, activation="relu", padding="same")
        ])
        self.encoder_1_pooling = MaxPool2D()
        self.center = Sequential([
            Conv2D(128, KERNEL_SIZE, activation="relu", padding="same"),
            Conv2D(128, KERNEL_SIZE, activation="relu", padding="same"),
            UpSampling2D((2, 2), interpolation="bilinear")
        ])
        self.decoder_0 = Sequential([
            Conv2D(64, KERNEL_SIZE, activation="relu", padding="same"),
            Conv2D(64, KERNEL_SIZE, activation="relu", padding="same"),
            UpSampling2D((2, 2), interpolation="bilinear")
        ])
        self.decoder_1 = Sequential([
            Conv2D(32, KERNEL_SIZE, activation="relu", padding="same"),
            Conv2D(3, KERNEL_SIZE, activation="relu", padding="same"),
        ])

    def __call__(self, current_x, previous_x):
        # axis 3, channel-wise
        flat_prev_channel_count = (self.frame_count - 1) * self.channels_per_frame
        desired_flat_shape = tf.concat(([-1], tf.shape(current_x)[1:][:-1], [flat_prev_channel_count]), axis=0)
        flat_prev_x = tf.reshape(previous_x, desired_flat_shape)
        x = tf.concat([current_x, flat_prev_x], axis=3)
        # encoder
        h_enc0_0 = self.encoder_0(x)
        h_enc0_1 = self.encoder_0_pooling(h_enc0_0)
        h_enc1_0 = self.encoder_1(h_enc0_1)
        h_enc1_1 = self.encoder_1_pooling(h_enc1_0)
        
        h_center = self.center(h_enc1_1)
        # concat along channel axis
        h_center = tf.concat([h_enc1_0, h_center], axis=3)
        h_dec0 = self.decoder_0(h_center)
        h_dec0 = tf.concat([h_enc0_0, h_dec0], axis=3)
        h_dec1 = self.decoder_1(h_dec0)

        return h_dec1