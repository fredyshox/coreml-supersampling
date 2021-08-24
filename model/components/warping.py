import tensorflow as tf

import tensorflow_addons as tfa

class BackwardWarp(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, image: tf.Tensor, flow: tf.Tensor):
        return tfa.image.dense_image_warp(image, flow)

class AccumulativeBackwardWarp(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.backward_warp = BackwardWarp()
    
    def __call__(self, image: tf.Tensor, flows: tf.Tensor):
        # flow is in shape: [batch, seq, height, width, channels]
        tf.assert_equal(flows.ndim, 5)

        _, seq_len, _, _, _ = tf.shape(flows)
        warped_image = image
        for index in range(seq_len):
            warped_image = self.backward_warp(warped_image, flows[:, index, :, :, :])
        
        return warped_image