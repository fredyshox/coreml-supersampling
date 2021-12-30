import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D

from model.components.zero_upsampling import ZeroUpsampling
from model.components.warping import BackwardWarp, AccumulativeBackwardWarp, AccumulativeBackwardWarpFast
from model.components.reconstruction import ReconstructionModule4X
from model.components.extraction import FeatureExtractionModule

class SuperSamplingModel(tf.keras.Model):
    def __init__(self, upsampling_factor, layer_config, upsize_type, warp_type, feature_extraction_enabled=True, frame_count=5):
        super().__init__()

        assert warp_type in ["single", "acc", "accfast"], "Invalid warp_type. Supported values: single, acc, accfast"

        self.frame_count = frame_count
        if feature_extraction_enabled:
            self.feature_extraction = FeatureExtractionModule()
        else:
            self.feature_extraction = None
        self.zero_upsampling = ZeroUpsampling(scale_factor=upsampling_factor)
        self.bilinear_upsampling = UpSampling2D(size=(upsampling_factor, upsampling_factor), interpolation='bilinear')
        if warp_type == "single":
            self.backward_warping = BackwardWarp()
        elif warp_type == "accfast":
            self.backward_warping = AccumulativeBackwardWarpFast()
        else:
            self.backward_warping = AccumulativeBackwardWarp()
        self.reconstruction = ReconstructionModule4X(
            frame_count=frame_count, layer_config=layer_config, upsize_type=upsize_type,
            channels_per_frame=12 if feature_extraction_enabled else 4, output_channels=1 # YUV mod
        )

    def compile(self, perceptual_loss, perceptual_loss_model, perceptual_loss_weight, *args, **kwargs):
        super(SuperSamplingModel, self).compile(*args, **kwargs)
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss_model = perceptual_loss_model
        self.perceptual_loss_model.trainable = False
        self.perceptual_loss_weight =  tf.convert_to_tensor(perceptual_loss_weight, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        # every thing in format [batch, seq, height, width, channels]
        rgb_frames = inputs["color"] # seq = frame_count
        depth_frames = inputs["depth"] # seq = frame_count
        motion_frames = inputs["motion"] # seq = frame_count - 1

        previous_frame_count = self.frame_count - 1

        rgbd_frames = tf.concat((rgb_frames, depth_frames), axis=4)
        desired_rgbd_shape = tf.concat(([-1], tf.shape(rgbd_frames)[-3:]), axis=0) # (-1, *tf.shape(rgbd_frames)[-3:])
        flat_rgbd_frames = tf.reshape(rgbd_frames, desired_rgbd_shape)
        if self.feature_extraction is not None:
            flat_features_frames = self.feature_extraction(flat_rgbd_frames)
        else:
            flat_features_frames = flat_rgbd_frames
        flat_upsampled_features = self.zero_upsampling(flat_features_frames)
        upsampled_features = tf.reshape(
            flat_upsampled_features, 
            tf.concat((tf.shape(rgbd_frames)[:2], tf.shape(flat_upsampled_features)[-3:]), axis=0)
        )

        tf.assert_equal(tf.shape(motion_frames)[1], previous_frame_count)
        desired_motion_vec_shape = tf.concat(([-1], tf.shape(motion_frames)[-3:]), axis=0) # [batch, seq, height, width, channels] -> [batch, height, width, channels]
        flat_motion_vectors = tf.reshape(motion_frames, desired_motion_vec_shape)
        flat_upsampled_motion_vectors = self.bilinear_upsampling(flat_motion_vectors)
        desired_upsampled_motion_vec_shape = tf.concat((tf.shape(motion_frames)[:2], tf.shape(flat_upsampled_motion_vectors)[-3:]), axis=0)
        upsampled_motion_vectors = tf.reshape(flat_upsampled_motion_vectors, desired_upsampled_motion_vec_shape)

        upsampled_previous_features = upsampled_features[:, :previous_frame_count, :, :, :]
        backward_warped_features = self.backward_warping(upsampled_previous_features, upsampled_motion_vectors)

        upsampled_current_features = upsampled_features[:, previous_frame_count, :, :, :]
        reconstructed_frame = self.reconstruction(upsampled_current_features, backward_warped_features)

        return reconstructed_frame
    
    @tf.function
    def train_step(self, data):
        inputs, targets = data
        assert len(inputs) == 3, "Inputs must consist of: rgb tensor, depth tensor, motion vec tensor"

        with tf.GradientTape() as tape:
            reconstructions = self(inputs, training=True)
            reconstructions_clipped = tf.clip_by_value(reconstructions, 0.0, 1.0)
            rgb_reconstructions = tf.image.yuv_to_rgb(reconstructions)
            rec_maps = self.perceptual_loss_model(rgb_reconstructions, training=False)
            target_maps = self.perceptual_loss_model(targets, training=False)
            p_loss = self.perceptual_loss(target_maps, rec_maps)
            loss = self.compiled_loss(
                targets, reconstructions_clipped,
                regularization_losses=[self.perceptual_loss_weight * p_loss] # regularization_losses - losses to be added to compiled loss
            )
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(targets, rgb_reconstructions)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        inputs, targets = data
        assert len(inputs) == 3, "Inputs must consist of: rgb tensor, depth tensor, motion vec tensor"

        reconstructions = self(inputs, training=False)
        reconstructions_clipped = tf.clip_by_value(reconstructions, 0.0, 1.0)
        rec_maps = self.perceptual_loss_model(reconstructions_clipped, training=False)
        target_maps = self.perceptual_loss_model(targets, training=False)
        p_loss = self.perceptual_loss(target_maps, rec_maps)

        self.compiled_loss(
            targets, reconstructions,
            regularization_losses=[self.perceptual_loss_weight * p_loss] # regularization_losses - losses to be added to compiled loss
        )
        self.compiled_metrics.update_state(targets, reconstructions_clipped)

        return {m.name: m.result() for m in self.metrics} 