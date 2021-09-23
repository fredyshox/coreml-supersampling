import tensorflow as tf

def psnr(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.psnr(y_pred, y, max_val=1.0)

def ssim(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    yuv_y = tf.image.rgb_to_yuv(y)
    yuv_y_pred = tf.image.rgb_to_yuv(y_pred)
    return tf.image.ssim(yuv_y_pred, yuv_y, max_val=1.0)