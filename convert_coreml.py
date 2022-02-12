import os
import sys 
import argparse

import coremltools as ct
import tensorflow as tf

from model.model import SuperSamplingModel
from model.loader import resolve_weights_uri


def main(args):
    model = SuperSamplingModel(
        upsampling_factor=args.scale_factor,
        layer_config=args.rec_layer_config,
        upsize_type=args.rec_upsize_type, 
        warp_type=args.warp_type,
        feature_extraction_enabled=not args.no_feature_extraction,
        frame_count=args.frame_count
    )
    if args.weights_path is not None:
        weights_file_path = resolve_weights_uri(args.weights_path)
        load_weights_no_dataset(model, weights_file_path, args.input_shape)
    
    output_parent_dir = os.path.dirname(args.output_path)
    output_path_no_ext, output_ext = os.path.splitext(args.output_path)
    if output_ext != "mlprogram":
        output_path = output_path_no_ext + ".mlprogram"
    else:
        output_path = args.output_path
        
    conversion_subject = model if args.full else model.reconstruction
    # model compilation only supported on macos
    compile = args.compile if sys.platform == "darwin" else False
    ct_model = ct.convert(
        conversion_subject, 
        convert_to="mlprogram", 
        skip_model_load=not compile
        # TODO package_dir for compiled mlmodelc
    )
    ct_model.save(output_path)


def load_weights_no_dataset(model, weight_path, frame_count, shape):
    dummy_data = {
        "color": tf.ones([1, frame_count, *shape[::-1], 3]),
        "depth": tf.ones([1, frame_count, *shape[::-1], 1]),
        "motion": tf.ones([1, frame_count-1, *shape[::-1], 2]),
    }
    _ = model(dummy_data)
    model.load_weights(weight_path)


def parse_args():
    def dims_type(s):
        try:
            dims = [int(ss) for ss in s.split("x")]
            if len(dims) != 2:
                raise ValueError()
            
            return tuple(dims)
        except ValueError:
            msg = "Invalid dimensions. Expected format: WIDTHxHEIGHT"
            raise argparse.ArgumentTypeError(msg)

    parser = argparse.ArgumentParser(description="Convert super sampling model to coreml format")
    parser.add_argument("--output-path", required=True, type=str, help="Output directory")
    parser.add_argument("--weights-path", type=str, help="Path to file with weights to load. Can be local file path or clearml task output: `clearml://<task-id>/[<model-index>]`")
    parser.add_argument("--compile", action="store_true", help="Compile mlmodelc using coremlc")
    parser.add_argument("--full", help="store_true",  help="Convert full model, instead of only reconstruction module")
    # model config
    parser.add_argument("--scale-factor", default=4, type=int, help="Super sampling target scale factor (should match dataset paths)")
    parser.add_argument("--frame-count", default=5, type=int, help="Observed frame sequence length")
    parser.add_argument("--rec-upsize-type", default="upsample", choices=["upsample", "deconv"], help="Reconstruction block upsampling type")
    parser.add_argument("--rec-layer-config", default="standard", choices=["standard", "fast", "ultrafast"], help="Reconstruction layer config")
    parser.add_argument("--warp-type", default="single", choices=["single", "acc", "accfast"], help="Backward warping type")
    parser.add_argument("--no-feature-extraction", default=False, help="Disable feature extraction")
    # input size
    parser.add_argument("--input-shape", default="1280x720", type=dims_type, help="Input width x height")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    main(args)