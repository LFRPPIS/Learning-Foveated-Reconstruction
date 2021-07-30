from glob import glob
import tensorflow as tf
import numpy as np
import os


class InputLoader:
    def __init__(self, ground_truth_path, input_path, real_path, test_input_folder):
        self.ground_truth_path = ground_truth_path
        self.input_path = input_path
        self.real_path = real_path
        self.test_input_folder = test_input_folder
        self.ground_truth_path = os.path.join(self.ground_truth_path, '')
        self.input_path = os.path.join(self.input_path, '')
        self.real_path = os.path.join(self.real_path, '')
        self.test_input_folder = os.path.join(self.test_input_folder, '')

    def create_image_dataset(self):
        all_image_paths = [os.path.basename(f) for f in glob(f'{self.ground_truth_path}*.png')]
        all_image_paths = [str(path) for path in list(all_image_paths)]
        n_images = len(all_image_paths)
        if n_images > 0:
            all_image_array = np.asarray(all_image_paths)
            path_ds = tf.data.Dataset.from_tensor_slices(all_image_array)
            path_ds = path_ds.shuffle(buffer_size=n_images)
            image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            image_ds = None

        all_test_image_paths = [os.path.basename(f) for f in glob(f'{self.test_input_folder}*.png')]
        all_test_image_paths = [str(path) for path in list(all_test_image_paths)]
        n_test_images = len(all_test_image_paths)
        if n_test_images > 0:
            all_test_image_array = np.asarray(all_test_image_paths)
            test_path_ds = tf.data.Dataset.from_tensor_slices(all_test_image_array)
            test_image_ds = test_path_ds.map(self.load_and_preprocess_test_image,
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            test_image_ds = None

        return n_images, image_ds, test_image_ds

    def load_and_preprocess_image(self, filename):
        ground_truth_path = self.ground_truth_path + filename
        input_name = filename + "_interpolated.png"
        mask_name = filename + "_mask.png"
        input_path = self.input_path + input_name
        mask_path = self.input_path + mask_name
        real_path = self.real_path + filename

        ground_truth_image = preprocess_image(tf.io.read_file(ground_truth_path))
        interpolated_image = preprocess_image(tf.io.read_file(input_path))
        mask_image = tf.io.read_file(mask_path)
        mask_image = tf.image.decode_png(mask_image, channels=1)
        mask_image = tf.cast(mask_image, tf.float32) / 255.0
        input_image = tf.concat([interpolated_image, mask_image], axis=2)
        real_image = preprocess_image(tf.io.read_file(real_path))

        all_images = (ground_truth_image, input_image, real_image)
        return all_images

    def load_and_preprocess_test_image(self, filename):
        interpolated_image = preprocess_image(tf.io.read_file(self.test_input_folder + filename))
        return interpolated_image


def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.dtypes.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image
