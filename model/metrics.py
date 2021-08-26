import tensorflow as tf

def psnr(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.psnr(y_pred, y, max_val=1.0)

def ssim(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: 
    return tf.image.ssim(y_pred, y, max_val=1.0)