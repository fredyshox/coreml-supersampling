#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from model.model import SuperSamplingModel
from model.loss import PerceptualLoss, SSIMLoss
from model.metrics import psnr, ssim
from model.dataset import RGBDMotionDataset
from model.utils import tf_minor_version_geq

UPSAMPLING_FACTOR = 4

def main(args):
    if args.debug:
        gpu_devices = tf.config.get_visible_devices("GPU")
        try: 
            if len(gpu_devices) != 0:
                tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        except Exception as ex:
            print(f"Debug exception: {ex}")
            pass
    if tf_minor_version_geq(4):
        tf.config.experimental.enable_tensor_float_32_execution(not args.no_tf32)
    if args.amp:
        if tf_minor_version_geq(6):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
    # create output dirs for checkpoint and logs
    os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    seq_overlap_mode = args.data_seq_overlap_mode
    if seq_overlap_mode.isnumeric():
        seq_overlap_mode = int(seq_overlap_mode)
    target_size = (
        args.patch_size[0] * UPSAMPLING_FACTOR, 
        args.patch_size[1] * UPSAMPLING_FACTOR
    )
    target_step = (
        args.patch_step[0] * UPSAMPLING_FACTOR,
        args.patch_step[1] * UPSAMPLING_FACTOR
    )
    dataset_factory = RGBDMotionDataset(
        args.data_root_dir, args.data_lr_subdir, args.data_hr_subdir,
        image_patch_size=args.patch_size, image_patch_step=args.patch_step,
        target_patch_size=target_size, target_patch_step=target_step
    )
    train_fraction = 1 - args.data_val_fraction
    train_dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode=seq_overlap_mode, split_fraction=train_fraction, use_keras_input_mapping=True
    ).batch(args.batch).shuffle(buffer_size=args.buffer_shuffle).prefetch(buffer_size=args.buffer_prefetch)
    val_dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode=seq_overlap_mode, split_fraction=train_fraction, take_top=True, use_keras_input_mapping=True
    ).batch(args.batch).shuffle(buffer_size=args.buffer_shuffle).prefetch(buffer_size=args.buffer_prefetch)

    model = SuperSamplingModel(upsize_type=args.rec_upsize_type, warp_type=args.warp_type)
    optimizer = Adam(learning_rate=args.lr)
    perceptual_model = perceptual_vgg_model(target_size)
    perceptual_loss = PerceptualLoss()
    ssim_loss = SSIMLoss()
    model.compile(
        perceptual_loss=perceptual_loss, 
        perceptual_loss_model=perceptual_model,
        perceptual_loss_weight=args.p_loss_weight,
        optimizer=optimizer,
        loss=ssim_loss,
        metrics=[psnr, ssim]
    )
    if args.weights_path is not None and len(args.weights_path) != 0:
        input_element_spec = train_dataset.element_spec[0]
        dummy_data = dict()
        for key in ["color", "depth", "motion"]:
            shape = input_element_spec[key].shape
            dummy_tensor = tf.ones([args.batch, *shape[-4:]])
            dummy_data[key] = dummy_tensor
        _ = model(dummy_data)
        model.load_weights(args.weights_path)
    model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        callbacks=[
            TensorBoard(log_dir=args.log_dir),
            ModelCheckpoint(filepath=args.checkpoint_dir, save_weights_only=True),
            EarlyStopping(patience=2)
        ],
        validation_data=val_dataset
    )


def perceptual_vgg_model(target_size):
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(*target_size, 3))
    custom_vgg_model = Model(vgg_model.input, [vgg_model.get_layer(layer).output for layer in ["block2_conv2", "block3_conv3"]])
    custom_vgg_model.trainable = False

    return custom_vgg_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train super sampling model")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--log-dir", default="logs", help="Dir to save logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints/model.weights.{epoch:02d}-{val_loss:.2f}.hdf5", help="Format/filepath to save model checkpoints")
    parser.add_argument("--batch", default=2, type=int, help="Batch size")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs")
    parser.add_argument("--p-loss-weight", default=0.1, type=float, help="Perceptual loss weight")
    parser.add_argument("--patch-size", default=[120, 120], action="store", type=int, nargs=2, help="Image patch size")
    parser.add_argument("--patch-step", default=[60, 60], action="store", type=int, nargs=2, help="Image patch step")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-val-fraction", default=0.1, type=float, help="Validation dataset fraciton")
    parser.add_argument("--data-seq-overlap-mode", default="all", help="Dataset frame sequence overlap strategory (all, none, [0-9])")
    parser.add_argument("--buffer-shuffle", default=128, type=int, help="Dataset shuffle buffer size")
    parser.add_argument("--buffer-prefetch", default=64, type=int, help="Dataset prefetch buffer size")
    parser.add_argument("--rec-upsize-type", default="upsample", choices=["upsample", "deconv"], help="Reconstruction block upsampling type")
    parser.add_argument("--rec-layer-config", default="standard", choices=["standard", "fast", "ultrafast"], help="Reconstruction layer config")
    parser.add_argument("--warp-type", default="single", choices=["single", "acc", "accfast"], help="Backward warping type")
    parser.add_argument("--weights-path", default=None, type=str, help="Path to file with weights to load (resume training)")
    parser.add_argument("--initial-epoch", default=0, type=int, help="Initial epoch (resume training)")
    parser.add_argument("--amp", action="store_true", help="Enable NVIDIA Automatic Mixed Precision")
    parser.add_argument("--no-tf32", action="store_true", help="Disable tensor float 32 support")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        print(f"Argparse arguments: {args}")
    main(args)