import tensorflow as tf

class ZeroUpsampling(tf.Module):
    def __init__(self, scale_factor, name=None):
        super().__init__(name=name)
        self.scale_factor = scale_factor
        self.scale_mat = tf.constant([1, scale_factor, scale_factor, 1], dtype=tf.int32)

    def __call__(self, x):
        input_shape = tf.shape(x)
        kernel = tf.ones([1, 1, input_shape[-1], input_shape[-1]])
        output_shape = input_shape * self.scale_mat
        return tf.nn.conv2d_transpose(x, kernel, output_shape, self.scale_factor, padding="VALID")