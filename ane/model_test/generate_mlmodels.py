import os
import sys
from collections import namedtuple

import coremltools as ct
import tensorflow as tf

from model.components.reconstruction import ReconstructionModule4X

from ane.tests.dummy_models import *


def get_test_models(rec_model, shape):
    return [
        CumsumModel(rec_model, axis=1),
        CumsumModel(rec_model, axis=3),
        CumsumModel(rec_model, axis=3, reverse=True),
        CustomCumsumModelAxis3(rec_model, shape),
        ArithmenticsWithBroadcastingModel(rec_model, shape, same_shape=True),
        ArithmenticsWithBroadcastingModel(rec_model, shape, scalar=True),
        ArithmenticsWithBroadcastingModel(rec_model, shape, broadcasting=True),
        MaxMinModel(rec_model),
        FloorModel(rec_model),
        GatherModel(rec_model, shape, axis=1),
        GatherModel(rec_model, shape, axis=3),
        SplitModel(rec_model, axis=1),
        SplitModel(rec_model, axis=3)
    ]


def main(args):
    rec_model = ReconstructionModule4X(3, "fast", "deconv", 12)
    input_shape = (1, 1280, 720, 36)

    # force model build
    print("===> Building reconstruction model")
    dummy_input = tf.ones(input_shape)
    _ = rec_model(dummy_input)

    print("===> Creating dummy mlmodels")
    for model in get_test_models(rec_model, input_shape):
        print(f"===> Converting {model.name} to coreml...")
        _ = model(dummy_input)
        ct_model = ct.convert(model, convert_to="mlprogram", skip_model_load=True)
        output_path = os.path.join(args.output_dir, f"{model.name}.mlpackage")
        ct_model.save(output_path)


def parse_args():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} outputdir")
        exit(1)

    Args = namedtuple("Args", ["output_dir"])
    args = Args(sys.argv[1])
    return args


if __name__=="__main__":
    args = parse_args()
    main(args)
