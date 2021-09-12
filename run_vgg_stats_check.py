import argparse
import tensorflow as tf
from tqdm import tqdm

from model.dataset import RGBDMotionDataset
from model.vgg import PerceptualFPVGG16

UPSAMPLING_FACTOR = 4


def online_welford_mean_var(gen):
    n = 0
    mean = 0
    M2 = 0
    for x in gen:
        n += 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + delta*(x - mean)
    
    variance = M2 / (n - 1)
    return (n, mean, variance)


def create_dataset(args):
    patch_size, patch_step = None, None
    target_size, target_step = None, None
    if args.patch_step is not None and args.patch_size is not None:
        patch_size = args.patch_size
        patch_step = args.patch_step
        target_size = (
            patch_size[0] * UPSAMPLING_FACTOR, 
            patch_size[1] * UPSAMPLING_FACTOR
        )
        target_step = (
            patch_step[0] * UPSAMPLING_FACTOR,
            patch_step[1] * UPSAMPLING_FACTOR
        )
    dataset_factory = RGBDMotionDataset(
        args.data_root_dir, args.data_lr_subdir, args.data_hr_subdir,
        image_patch_size=args.patch_size, image_patch_step=args.patch_step,
        target_patch_size=target_size, target_patch_step=target_step
    )
    dataset = dataset_factory.tf_dataset(
        seq_frame_overlap_mode="all", 
        use_keras_input_mapping=True, 
        create_patches=(patch_size is not None)
    )
    dataset = dataset.batch(args.batch)
    if args.data_limit is not None:
        dataset = dataset.take(args.data_limit)
    dataset = dataset.shuffle(args.buffer_shuffle).prefetch(args.buffer_prefetch)

    return dataset


def log_clearml(data):
    header = ["layer", "n", "mean", "variance", "stardard deviation"]
    rows = [header] + [list(row) for row in data]
    clearml.Logger.current_logger().report_table(
        "VGG activation stats",
        "n/mean/var/stddev",
        table_plot=rows
    )


def main(args):
    dataset = create_dataset(args)

    input_element_spec = dataset.element_spec[1]
    input_shape = input_element_spec.shape[-3:]
    results = []
    for layer in args.vgg_layers:
        print(f"==> Stats for layer: {layer}")
        model = PerceptualFPVGG16("imagenet", input_shape, [layer])
        def generator():
            for batch in tqdm(iter(dataset), desc="Calculate dataset stats", total=args.data_limit):
                _, targets = batch
                activations = model(targets)
                activations_mean = tf.reduce_mean(activations)
                yield activations_mean
        n, mean, variance = online_welford_mean_var(generator())
        stddev = tf.sqrt(variance)
        print(f"==> n: {n}, mean: {mean}, variance: {variance}, stddev: {stddev}")
        results.append((layer, n, mean, variance, stddev))
    
    if args.clearml:
        import clearml
        log_clearml(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset distribution")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root dir")
    parser.add_argument("--data-lr-subdir", required=True, help="Dataset low-res subdir")
    parser.add_argument("--data-hr-subdir", required=True, help="Dataset high-res subdir")
    parser.add_argument("--data-limit", default=None, type=int, help="Dataset sample limit")
    parser.add_argument("--patch-size", default=None, action="store", type=int, nargs=2, help="Image patch size")
    parser.add_argument("--patch-step", default=None, action="store", type=int, nargs=2, help="Image patch step")
    parser.add_argument("--vgg-layers", required=True, action="store", type=str, nargs="+", help="VGG layers to use in perceptual loss")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    parser.add_argument("--buffer-shuffle", default=128, type=int, help="Dataset shuffle buffer size")
    parser.add_argument("--buffer-prefetch", default=64, type=int, help="Dataset prefetch buffer size")
    parser.add_argument("--clearml", action="store_true", help="Use clearml for logging")

    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = parse_args()
    main(args)