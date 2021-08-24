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
        for pred_pred, pred_true in zip(y_pred_maps, y_true_maps):
            perceptual_loss += tf.reduce_mean(
                tf.square(tf.subtract(pred_pred, pred_true)),
                axis=tf.range(1, 4)
            )

        return perceptual_loss