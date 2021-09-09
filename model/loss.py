import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction

class SSIMLoss(Loss):
    def __init__(self, reduction=Reduction.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
    
    def call(self, y_true, y_pred):
        ssim_loss = 1 - tf.image.ssim(y_pred, y_true, 1.0)
        return ssim_loss

class PerceptualLoss(Loss):
    def __init__(self, reduction=Reduction.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true_maps, y_pred_maps):
        perceptual_loss = 0
        if tf.is_tensor(y_pred_maps) and tf.is_tensor(y_true_maps): # tensor input
            perceptual_loss = self._layer_p_loss(y_pred_maps, y_true_maps)
        else: # iterable multi input - probably should check if it's really iterable tho
            for pred_pred, pred_true in zip(y_pred_maps, y_true_maps):
                perceptual_loss += self._layer_p_loss(pred_pred, pred_true)

        return perceptual_loss

    @tf.function
    def _layer_p_loss(self, pred_map, true_map):
        return tf.reduce_mean(
            tf.square(tf.subtract(pred_map, true_map)),
            axis=tf.range(1, 4)
        )