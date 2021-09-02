import os
from glob import glob
import tensorflow as tf
import tensorflow_io as tfio

from model.utils import tf_minor_version_geq

COLOR_ID = "COLOR"
DEPTH_ID = "DEPTH"
MOTION_ID = "MOTIONVECTORS"

class RGBDMotionDataset:
    def __init__(
        self, root_dir, lr_subdir, hr_subdir, 
        frames_per_rec=60, 
        frames_per_sample=5, 
        image_patch_size=(120, 120), 
        image_patch_step=(60, 60), 
        target_patch_size=(480, 480),
        target_patch_step=(240, 240)
    ):
        assert(len(image_patch_size) == 2)
        self.image_patch_size = image_patch_size
        self.image_patch_step = image_patch_step
        self.target_patch_size = target_patch_size
        self.target_patch_step = target_patch_step
        self.frames_per_rec = frames_per_rec
        self.frames_per_sample = frames_per_sample
        self.root_dir = root_dir
        self.lr_subdir = lr_subdir
        self.hr_subdir = hr_subdir

    def _image_seq_map_func(self, rec_subdir, indices):
        rec_name = os.path.basename(rec_subdir)
        lr_color_paths = [
            os.path.join(self.root_dir, rec_subdir, self.lr_subdir, f"{rec_name}.{self.lr_subdir}.{COLOR_ID}.{i}.png") 
            for i in indices
        ]
        lr_depth_paths = [
            os.path.join(self.root_dir, rec_subdir, self.lr_subdir, f"{rec_name}.{self.lr_subdir}.{DEPTH_ID}.{i}.exr") 
            for i in indices
        ]
        lr_motion_paths = [
            os.path.join(self.root_dir, rec_subdir, self.lr_subdir, f"{rec_name}.{self.lr_subdir}.{MOTION_ID}.{i}.exr") 
            for i in indices[1:]
        ]
        hr_color_path = os.path.join(self.root_dir, rec_subdir, self.hr_subdir, f"{rec_name}.{self.hr_subdir}.{COLOR_ID}.{indices[-1]}.png") 
        lr_color_contents = [tf.io.read_file(path) for path in lr_color_paths]
        lr_depth_contents = [tf.io.read_file(path) for path in lr_depth_paths]
        lr_motion_contents = [tf.io.read_file(path) for path in lr_motion_paths]
        hr_color_content = tf.io.read_file(hr_color_path)

        lr_color_tensors = [
            tf.image.convert_image_dtype(tf.io.decode_png(content, channels=3), tf.float32)
            for content in lr_color_contents
        ]
        lr_depth_tensors = [
            tf.expand_dims(
                tf.cast(tfio.experimental.image.decode_exr(content, 0, "R", tf.float16), tf.float32),
                axis=2
            )
            for content in lr_depth_contents
        ]
        lr_motion_tensors = [
            tf.stack(
                [tf.cast(tfio.experimental.image.decode_exr(content, 0, "R", tf.float16), tf.float32),
                 tf.cast(tfio.experimental.image.decode_exr(content, 0, "G", tf.float16), tf.float32)],
                axis=2
            )
            for content in lr_motion_contents
        ]
        hr_color_tensor = tf.expand_dims(
            tf.image.convert_image_dtype(tf.io.decode_png(hr_color_content, channels=3), tf.float32),
            axis=0
        )

        x = [
            self._create_image_patches(tf.stack(lr_color_tensors)), 
            self._create_image_patches(tf.stack(lr_depth_tensors)), 
            self._create_image_patches(tf.stack(lr_motion_tensors))
        ]
        y = tf.squeeze(self._create_image_patches(hr_color_tensor, mode='target'), axis=[1])

        return (*x, y)

    def _create_image_patches(self, tensors, mode="input"):
        if mode == "input":
            kernel = (1, self.image_patch_size[0], self.image_patch_size[1], 1)
            strides = (1, self.image_patch_step[0], self.image_patch_step[1], 1)
        elif mode == "target":
            kernel = (1, self.target_patch_size[0], self.target_patch_size[1], 1)
            strides = (1, self.target_patch_step[0], self.target_patch_step[1], 1)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        patches = tf.image.extract_patches(tensors, kernel, strides, (1,1,1,1), 'VALID')
        new_shape = (tensors.shape[0], -1, kernel[1], kernel[2], tensors.shape[3])
        patches = tf.reshape(patches, new_shape)
        patches = tf.transpose(patches, [1, 0, 2, 3, 4])
        
        return patches # shape is: [patch_count, batch_size, height, width, channels]

    def _frame_indices(self, overlap_mode):
        indices = []
        step = None
        if isinstance(overlap_mode, str):
            step = 1 if overlap_mode == "all" else self.frames_per_sample # none
        elif isinstance(overlap_mode, int):
            step = overlap_mode

        for i in range(self.frames_per_sample, self.frames_per_rec + 1, step):
            indices.append(list(range(i - self.frames_per_sample, i)))

        return indices

    def tf_dataset(self, seq_frame_overlap_mode="all", split_fraction=None, take_top=False, use_keras_input_mapping=False):
        if isinstance(seq_frame_overlap_mode, str):
            assert seq_frame_overlap_mode in ["all", "none"], "seq_frame_overlap_mode possible string values overlap_all, overlap_none"
        elif isinstance(seq_frame_overlap_mode, int):
            assert seq_frame_overlap_mode in range(1, self.frames_per_sample + 1), "seq_frame_overlap_mode integer value must be in range 1...frames_per_sample"

        rec_paths = glob(os.path.join(self.root_dir, "*"))
        rec_paths = [path for path in rec_paths if os.path.isdir(path)]
        if split_fraction is not None:
            if not take_top:
                rec_paths = rec_paths[:int(split_fraction * len(rec_paths))]
            else:
                rec_paths = rec_paths[int(split_fraction * len(rec_paths)):]

        if len(rec_paths) == 0:
            raise ValueError(f"Found 0 recordings in {self.root_dir}!")

        def image_generator():
            for path in rec_paths:
                for ids in self._frame_indices(seq_frame_overlap_mode):
                    samples = self._image_seq_map_func(path, ids)
                    yield samples
        
        if tf_minor_version_geq(5):
            dataset = tf.data.Dataset.from_generator(
                image_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(None, self.frames_per_sample, self.image_patch_size[0], self.image_patch_size[1], 3), dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(None, self.frames_per_sample, self.image_patch_size[0], self.image_patch_size[1], 1), dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(None, self.frames_per_sample - 1, self.image_patch_size[0], self.image_patch_size[1], 2), dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(None, self.target_patch_size[0], self.target_patch_size[1], 3), dtype=tf.float32
                    )
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                image_generator,
                output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                output_shapes=(
                    (None, self.frames_per_sample, self.image_patch_size[0], self.image_patch_size[1], 3),
                    (None, self.frames_per_sample, self.image_patch_size[0], self.image_patch_size[1], 1),
                    (None, self.frames_per_sample - 1, self.image_patch_size[0], self.image_patch_size[1], 2),
                    (None, self.target_patch_size[0], self.target_patch_size[1], 3)
                )
            )

        dataset = dataset.flat_map(lambda rgb, d, mv, y: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(rgb),
            tf.data.Dataset.from_tensor_slices(d),
            tf.data.Dataset.from_tensor_slices(mv),
            tf.data.Dataset.from_tensor_slices(y)
        )))
        if use_keras_input_mapping:
            # keras fit will accept input in format [inputs, targets]
            dataset = dataset.map(lambda rgb, d, mv, y: ({"color": rgb, "depth": d, "motion": mv}, y))

        return dataset