#!/usr/bin/env python3
#
# Script for debugging preprocessing 
#

import os
import argparse

import tensorflow as tf
from tqdm import tqdm

from model.model import PreprocessingModel
from model.dataset import RGBDMotionDataset


def main(args):
    dataset_factory = RGBDMotionDataset(
        args.data_root_dir, args.data_lr_subdir, args.data_hr_subdir, 
        frames_per_sample=args.frame_count
    )
    dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode="all", 
        use_keras_input_mapping=False, 
        create_patches=False,
        include_paths=True
    ).batch(1)

    model = PreprocessingModel(args.scale_factor, args.warp_type, args.frame_count)

    for rgb, _, mv, _, fp in tqdm(dataset, desc="Enumerating dataset"):
        _, previous = model(rgb, mv)
        for i, (image, filepath) in enumerate(zip(previous[0], fp[0][:-1])):
            u8_image = tf.image.convert_image_dtype(image, tf.uint8)
            image_data = tf.io.encode_png(u8_image)
            filename_noext, _ = os.path.splitext(os.path.basename(filepath.numpy().decode()))
            output_filepath = os.path.join(args.output_dir, f"{filename_noext}+{args.frame_count-1-i}.png")
            tf.io.write_file(output_filepath, image_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Perform preprocessing (upsample+warp)")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--scale-factor", default=2, type=int, help="Super sampling target scale factor (should match dataset paths)")
    parser.add_argument("--frame-count", default=3, type=int, help="Observed frame sequence length")
    parser.add_argument("--warp-type", default="accfast", choices=["single", "acc", "accfast"], help="Backward warping type")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    main(args)