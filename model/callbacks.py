from clearml.datasets import dataset
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class DebugSamplesCallback(Callback):
    def __init__(self, log_dir, dataset, n_epoch=1, n_images_per_epoch=4):
        super().__init__()

        self.dataset = dataset
        self.iterator = iter(dataset)
        self.n_epoch = n_epoch
        self.n_images_per_epoch = n_images_per_epoch
        self.file_writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs, fail_on_exhausted_dataset=False):
        if epoch % self.n_epoch != 0:
            return
        
        try:
            inputs, _ = next(self.iterator)
            preds = self.model(inputs)
            rgb_preds = tf.image.yuv_to_rgb(preds)
            with self.file_writer.as_default():
                tf.summary.image("Reconstructions", rgb_preds, step=epoch, max_outputs=self.n_images_per_epoch)
                self.file_writer.flush()
        except StopIteration:
            if fail_on_exhausted_dataset:
                raise ValueError("Samples dataset exhausted! Probably is empty!")

            print(f"===> {self}: reseting dataset iterator")
            self.iterator = iter(dataset)
            self.on_epoch_end(epoch, logs, fail_on_exhausted_dataset=True)        