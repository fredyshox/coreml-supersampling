import tensorflow as tf


def _get_dim(x, idx):
    if x.shape.ndims is None:
        return tf.shape(x)[idx]
    return x.shape[idx] or tf.shape(x)[idx]


class ZeroUpsampling(tf.Module):
    def __init__(self, scale_factor, name=None):
        super().__init__(name=name)
        self.scale_factor = scale_factor

    def __call__(self, x):
        height, width, channels = (
            _get_dim(x, 1),
            _get_dim(x, 2),
            _get_dim(x, 3),
        )

        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        doubled_grid_yx = tf.stack([grid_y, grid_x], axis=2)*self.scale_factor
        output_shape = tf.constant([height*self.scale_factor, width*self.scale_factor, channels])
        return tf.map_fn(lambda i: tf.scatter_nd(doubled_grid_yx, i, output_shape), elems=x)
