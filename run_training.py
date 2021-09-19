#!/usr/bin/env python3

import os
import re
import argparse
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError

from model.model import SuperSamplingModel
from model.loss import PerceptualLossMSE, YUV_MixL2SSIMLoss, YUV_SSIMLoss, YUV_MixL1SSIMLoss
from model.metrics import psnr, ssim
from model.dataset import RGBDMotionDataset
from model.vgg import PerceptualFPVGG16
from model.callbacks import DebugSamplesCallback
from model.utils import tf_minor_version_geq

UPSAMPLING_FACTOR = 4
DEFAULT_VGG_LOSS_LAYERS = ["block2_conv2", "block3_conv3"]
DEFAULT_MIX_LOSS_SSIM_WEIGHT = 0.8


def create_datasets(args):
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
        seq_frame_overlap_mode=seq_overlap_mode, 
        split_fraction=train_fraction, 
        use_keras_input_mapping=True
    )
    train_dataset = train_dataset.batch(args.batch) \
        .shuffle(buffer_size=args.buffer_shuffle, seed=args.seed) \
        .prefetch(buffer_size=args.buffer_prefetch)
    val_dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode=seq_overlap_mode, 
        split_fraction=train_fraction, 
        take_top=True, 
        use_keras_input_mapping=True
    )
    val_dataset = val_dataset.batch(args.batch) \
        .shuffle(buffer_size=args.buffer_shuffle, seed=args.seed) \
        .prefetch(buffer_size=args.buffer_prefetch)

    return (train_dataset, val_dataset, target_size, target_step)


def toggle_global_options(args):
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
    if args.seed is not None:
        tf.random.set_seed(args.seed)


def create_base_loss(name):
    if name == "l1":
        return MeanAbsoluteError()
    elif name == "l2":
        return MeanSquaredError()
    elif name == "ssim":
        return YUV_SSIMLoss()
    
    name_re = r"([a-z0-9]+)\+([a-z0-9]+)(:([01]\.[0-9]+))?"
    match = re.fullmatch(name_re, name)
    if match is None:
        raise ValueError(f"Unsupported loss function name: {name}")
    
    l_names = [match.group(1), match.group(2)]
    weight = DEFAULT_MIX_LOSS_SSIM_WEIGHT
    if match.group(4) is not None:
        weight = float(match.group(4))
    
    if "ssim" in l_names and "l1" in l_names:
        return YUV_MixL1SSIMLoss(weight)
    elif "ssim" in l_names and "l2" in l_names:
        return YUV_MixL2SSIMLoss(weight)
    else: # TODO Allow more combinations, or every possible
        raise ValueError(f"Unsupported loss function combinaion: {l_names}")


def create_or_load_model(args, dataset, target_size):
    model = SuperSamplingModel(
        layer_config=args.rec_layer_config, 
        upsize_type=args.rec_upsize_type, 
        warp_type=args.warp_type
    )
    optimizer = Adam(learning_rate=args.lr)
    perceptual_model = PerceptualFPVGG16(
        weights="imagenet",
        input_shape=(*target_size, 3),
        output_layer_names=args.vgg_layers,
        loc=args.vgg_norm_loc,
        scale=args.vgg_norm_scale
    )
    perceptual_loss = PerceptualLossMSE()
    loss = create_base_loss(args.loss)
    model.compile(
        perceptual_loss=perceptual_loss, 
        perceptual_loss_model=perceptual_model,
        perceptual_loss_weight=args.p_loss_weight,
        optimizer=optimizer,
        loss=loss,
        metrics=[psnr, ssim]
    )

    # load model if weights path is provided
    if args.weights_path is not None and len(args.weights_path) != 0:
        # call model with some dummy data to force tf into building it
        input_element_spec = dataset.element_spec[0]
        dummy_data = dict()
        for key in ["color", "depth", "motion"]:
            shape = input_element_spec[key].shape
            dummy_tensor = tf.ones([args.batch, *shape[-4:]])
            dummy_data[key] = dummy_tensor
        _ = model(dummy_data)
        model.load_weights(args.weights_path)

    return model 


def create_callbacks(args, val_dataset):
    callbacks = [
        TensorBoard(log_dir=args.log_dir),
        ModelCheckpoint(filepath=args.checkpoint_dir, save_weights_only=True),
        DebugSamplesCallback(log_dir=args.log_dir, dataset=val_dataset)
    ]
    if args.lr_decay is not None:
        def drop_step_decay(epoch):
            initial_lr = args.lr
            drop = args.lr_decay
            epochs_drop = 5
            lr = initial_lr * tf.math.pow(drop, tf.math.floor((1+epoch)/epochs_drop))
            return lr
        callbacks.append(
            LearningRateScheduler(drop_step_decay)
        )

    return callbacks


def main(args):
    toggle_global_options(args)
    
    # create output dirs for checkpoint and logs
    os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    train_dataset, val_dataset, target_size, _ = create_datasets(args)
    model = create_or_load_model(args, train_dataset, target_size)
    callbacks = create_callbacks(args, val_dataset)
    
    model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        callbacks=callbacks,
        validation_data=val_dataset
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train super sampling model")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--lr-decay", default=None, type=float, help="Learning rate decay")
    parser.add_argument("--log-dir", default="logs", help="Dir to save logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints/model.weights.{epoch:02d}-{val_loss:.2f}.hdf5", help="Format/filepath to save model checkpoints")
    parser.add_argument("--batch", default=2, type=int, help="Batch size")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs")
    parser.add_argument("--loss", default="ssim", help="Base loss function (options: l1, l2, ssim, ssim+l1, ssim+l2)")
    parser.add_argument("--p-loss-weight", default=0.1, type=float, help="Perceptual loss weight (0.0 if should not be used)")
    parser.add_argument("--patch-size", default=[120, 120], action="store", type=int, nargs=2, help="Image patch size")
    parser.add_argument("--patch-step", default=[60, 60], action="store", type=int, nargs=2, help="Image patch step")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-val-fraction", default=0.1, type=float, help="Validation dataset fraction")
    parser.add_argument("--data-seq-overlap-mode", default="all", help="Dataset frame sequence overlap strategory (all, none, [0-9])")
    parser.add_argument("--buffer-shuffle", default=128, type=int, help="Dataset shuffle buffer size")
    parser.add_argument("--buffer-prefetch", default=64, type=int, help="Dataset prefetch buffer size")
    parser.add_argument("--rec-upsize-type", default="upsample", choices=["upsample", "deconv"], help="Reconstruction block upsampling type")
    parser.add_argument("--rec-layer-config", default="standard", choices=["standard", "fast", "ultrafast"], help="Reconstruction layer config")
    parser.add_argument("--warp-type", default="single", choices=["single", "acc", "accfast"], help="Backward warping type")
    parser.add_argument("--vgg-layers", default=DEFAULT_VGG_LOSS_LAYERS, action="store", type=str, nargs="+", help="VGG layers to use in perceptual loss")
    parser.add_argument("--vgg-norm-loc", default=0.0, type=float, help="Mean value used for vgg activation standardization (use 0.0 to disable)")
    parser.add_argument("--vgg-norm-scale", default=1.0, type=float, help="Standard deviation used for vgg activation standardization (use 1.0 to disable)")
    parser.add_argument("--weights-path", default=None, type=str, help="Path to file with weights to load (resume training)")
    parser.add_argument("--initial-epoch", default=0, type=int, help="Initial epoch (resume training)")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
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