import tensorflow as tf

import tensorflow_addons as tfa

# internal module for 4d warp operation
class _BackwardWarpFlat(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, image: tf.Tensor, flow: tf.Tensor):
        tf.assert_equal(tf.rank(image), 4)
        tf.assert_equal(tf.rank(flow), 4)

        return tfa.image.dense_image_warp(image, flow)


class BackwardWarp(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.backward_warp = _BackwardWarpFlat()

    def __call__(self, images: tf.Tensor, flows: tf.Tensor):
        # flow is in shape: [batch, seq, height, width, channels]
        tf.assert_equal(tf.rank(flows), 5)
        tf.assert_equal(tf.rank(images), 5)
        tf.assert_equal(tf.shape(images)[:-1], tf.shape(flows)[:-1])

        desired_flat_image_shape = tf.concat(([-1], tf.shape(images)[-3:]), axis=0)
        desired_flat_flows_shape = tf.concat(([-1], tf.shape(flows)[-3:]), axis=0)
        flat_images = tf.reshape(images, desired_flat_image_shape)
        flat_flows = tf.reshape(flows, desired_flat_flows_shape)
        warped_images_flat = self.backward_warp(flat_images, flat_flows)

        # back to 5d shape
        warped_images = tf.reshape(warped_images_flat, tf.shape(images))
        return warped_images


class AccumulativeBackwardWarp(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.backward_warp = _BackwardWarpFlat()
    
    def __call__(self, images: tf.Tensor, flows: tf.Tensor):
        # flow is in shape: [batch, seq, height, width, channels]
        tf.assert_equal(tf.rank(flows), 5)
        tf.assert_equal(tf.rank(images), 5)
        tf.assert_equal(tf.shape(images)[:-1], tf.shape(flows)[:-1])

        seq_len = tf.shape(flows)[1]

        warped_images = tf.unstack(images, axis=1)
        for img_index, image in enumerate(warped_images):
            for flow_index in tf.range(img_index, seq_len):
                image = self.backward_warp(image, flows[:, flow_index, :, :, :])
            warped_images[img_index] = image
        warped_image = tf.stack(warped_images, axis=1)
        
        return warped_image


class AccumulativeBackwardWarpFast(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.backward_warp = _BackwardWarpFlat()
    
    def __call__(self, images: tf.Tensor, flows: tf.Tensor):
        # image/flow is in shape: [batch, seq, height, width, channels]
        tf.assert_equal(tf.rank(flows), 5)
        tf.assert_equal(tf.rank(images), 5)
        tf.assert_equal(tf.shape(images)[:-1], tf.shape(flows)[:-1])

        # cumulative sums: flow1, flow1 + flow2, flow1 + flow2 + flow3...
        flow_sums = tf.cumsum(flows, axis=1, reverse=True)
        # reshape flat for vectorization
        desired_image_shape = tf.concat(([-1], tf.shape(images)[-3:]), axis=0)
        desired_flow_shape = tf.concat(([-1], tf.shape(flow_sums)[-3:]), axis=0)
        images_flat = tf.reshape(images, desired_image_shape)
        flow_sums_flat = tf.reshape(flow_sums, desired_flow_shape)
        warped_images_flat = self.backward_warp(images_flat, flow_sums_flat)
        
        # back to 5d shape
        warped_images = tf.reshape(warped_images_flat, tf.shape(images))
        return warped_images