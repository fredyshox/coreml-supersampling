#!/usr/bin/env python3 

import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D, InputLayer

from model.dataset import RGBDMotionDataset
from model.metrics import psnr, ssim

UPSAMPLING_FACTOR = 4


def bilinear_baseline(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(UpSampling2D(size=(UPSAMPLING_FACTOR, UPSAMPLING_FACTOR), interpolation="bilinear"))
    return model


def log_stdout(train_results, val_results, model_name):
    train_results_str = ", ".join([f"{key}: {value}" for key, value in train_results.items()])
    val_results_str = ", ".join([f"{key}: {value}" for key, value in val_results.items()])
    print(f"===> Model {model_name} (train): {train_results_str}")
    print(f"===> Model {model_name} (val):   {val_results_str}")


def log_clearml(train_results, val_results, model_name):
    import clearml
    keys = list(train_results.keys())
    headers = ["dataset"] + keys
    rows = [
        headers,
        ["train"] + [float(train_results[key]) for key in keys],
        ["val"] + [float(val_results[key]) for key in keys]
    ]
    clearml.Logger.current_logger().report_table(
        "Baselines",
        model_name,
        table_plot=rows
    )


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
        seq_frame_overlap_mode=seq_overlap_mode, split_fraction=train_fraction, use_keras_input_mapping=True
    )
    train_dataset = train_dataset.map(lambda d, t: (d["color"][-1, :, :, :], t))
    train_dataset = train_dataset.batch(args.batch)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_shuffle).prefetch(buffer_size=args.buffer_prefetch)
    val_dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode=seq_overlap_mode, split_fraction=train_fraction, take_top=True, use_keras_input_mapping=True
    )
    val_dataset = val_dataset.map(lambda d, t: (d["color"][-1, :, :, :], t))
    val_dataset = val_dataset.batch(args.batch)
    val_dataset = val_dataset.shuffle(buffer_size=args.buffer_shuffle).prefetch(buffer_size=args.buffer_prefetch)

    return train_dataset, val_dataset


def main(args):
    train_dataset, val_dataset = create_datasets(args)

    model = bilinear_baseline((args.patch_size[0], args.patch_size[1], 3))
    model.compile(metrics=[psnr, ssim])
    
    train_results = model.evaluate(train_dataset, return_dict=True)
    val_results = model.evaluate(val_dataset, return_dict=True)

    log_stdout(train_results, val_results, "bilinear-upsampling")
    if args.clearml:
        log_clearml(train_results, val_results, "bilinear-upsampling")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baselines")
    parser.add_argument("--batch", default=2, type=int, help="Batch size")
    parser.add_argument("--patch-size", default=[120, 120], action="store", type=int, nargs=2, help="Image patch size")
    parser.add_argument("--patch-step", default=[60, 60], action="store", type=int, nargs=2, help="Image patch step")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-val-fraction", default=0.1, type=float, help="Validation dataset fraciton")
    parser.add_argument("--data-seq-overlap-mode", default="all", help="Dataset frame sequence overlap strategory (all, none, [0-9])")
    parser.add_argument("--buffer-shuffle", default=128, type=int, help="Dataset shuffle buffer size")
    parser.add_argument("--buffer-prefetch", default=64, type=int, help="Dataset prefetch buffer size")
    parser.add_argument("--clearml", action="store_true", help="Use clearml storage/debug samples mechanism")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    main(args)