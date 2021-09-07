#!/usr/bin/env python3

import os
import argparse
import random
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tqdm import tqdm

from model.model import SuperSamplingModel
from model.dataset import RGBDMotionDataset

UPSAMPLING_FACTOR = 4
DEBUG_SAMPLE_COUNT = 8


def main(args):
    dataset_factory = RGBDMotionDataset(
        args.data_root_dir, args.data_lr_subdir, args.data_hr_subdir
    )
    dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode="all", use_keras_input_mapping=True, create_patches=False
    ).batch(args.batch).take(args.data_limit).prefetch(args.buffer_prefetch)

    model = SuperSamplingModel(
        layer_config=args.rec_layer_config,
        upsize_type=args.rec_upsize_type, 
        warp_type=args.warp_type
    )
    load_weights(model, args.weights_path, dataset)
    predictions = model.predict(dataset, verbose=1)

    images_path = os.path.join(args.output_dir, "reconstructions")
    os.makedirs(images_path, exist_ok=True)
    for i, image in tqdm(enumerate(predictions), desc="Saving predictions"):
        image_path = os.path.join(images_path, f"{i}_{UPSAMPLING_FACTOR}x.png")
        u8_image = tf.image.convert_image_dtype(image, tf.uint8)
        image_data = tf.io.encode_png(u8_image)
        tf.io.write_file(image_path, image_data)
    
    if args.clearml:
        import clearml
        task = clearml.Task.current_task()
        task.upload_artifact("predictions", artifact_object=image_path)
        logger = clearml.Logger.current_logger()
        debub_sample_filenames = random.shuffle(os.listdir(images_path))[:DEBUG_SAMPLE_COUNT]
        for filename in debub_sample_filenames:
            sample_path = os.path.join(images_path, filename)
            logger.report_image("reconstruction", f"Image {filename}", local_path=sample_path)
  

def load_weights(model, weight_path, dataset):
    input_element_spec = dataset.element_spec[0]
    dummy_data = dict()
    for key in ["color", "depth", "motion"]:
        shape = input_element_spec[key].shape
        dummy_tensor = tf.ones([1, *shape[-4:]])
        dummy_data[key] = dummy_tensor
    _ = model(dummy_data)
    model.load_weights(weight_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using super sampling model")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-limit", default=None, type=int, help="Dataset sample limit")
    parser.add_argument("--weights-path", required=True, type=str, help="Path to file with weights to load (resume training)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    parser.add_argument("--buffer-prefetch", default=64, type=int, help="Dataset prefetch buffer size")
    parser.add_argument("--rec-upsize-type", default="upsample", choices=["upsample", "deconv"], help="Reconstruction block upsampling type")
    parser.add_argument("--rec-layer-config", default="standard", choices=["standard", "fast", "ultrafast"], help="Reconstruction layer config")
    parser.add_argument("--warp-type", default="single", choices=["single", "acc", "accfast"], help="Backward warping type")
    parser.add_argument("--clearml", default=False, action="store_true", help="Use clearml storage/debug samples mechanism")

    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = parse_args()
    main(args)