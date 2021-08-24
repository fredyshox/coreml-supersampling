#!/usr/bin/env python3

import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from model.model import SuperSamplingModel
from model.loss import PerceptualLoss, SSIMLoss
from model.metrics import psnr, ssim
from model.dataset import RGBDMotionDataset

UPSAMPLING_FACTOR = 4

def main(args):
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
        split_fraction=train_fraction, use_keras_input_mapping=True
    ).batch(args.batch)
    val_dataset = dataset_factory.tf_dataset(
        split_fraction=train_fraction, take_top=True, use_keras_input_mapping=True
    ).batch(args.batch)

    model = SuperSamplingModel()
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
    ModelCheckpoint()
    model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=[
            TensorBoard(log_dir=args.log_dir),
            ModelCheckpoint(filepath=args.checkpoint_dir, save_best_only=True),
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
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--log-dir", default="logs", help="Dir to save logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints/model.hdf5", help="Dir to save model checkpoints")
    parser.add_argument("--batch", default=2, type=int, help="Batch size")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs")
    parser.add_argument("--p-loss-weight", default=0.1, type=float, help="Perceptual loss weight")
    parser.add_argument("--patch-size", default=[120, 120], action="store", type=int, nargs=2, help="Image patch size")
    parser.add_argument("--patch-step", default=[60, 60], action="store", type=int, nargs=2, help="Image patch step")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-val-fraction", default=0.1, type=float, help="Validation dataset fraciton")


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)