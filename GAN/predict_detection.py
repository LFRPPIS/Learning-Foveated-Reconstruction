from argparse import ArgumentParser

import pandas as pd
import numpy as np
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


def make_vgg():
    vgg = VGG19(weights="imagenet", input_shape=(None, None, 3), include_top=False)
    vgg.outputs = [
        vgg.get_layer('block1_conv1').output,
        vgg.get_layer('block1_conv2').output,
        vgg.get_layer('block1_pool').output,
        vgg.get_layer('block2_conv1').output,
        vgg.get_layer('block2_conv2').output,
        vgg.get_layer('block2_pool').output,
        vgg.get_layer('block3_conv1').output,
        vgg.get_layer('block3_conv2').output,
        vgg.get_layer('block3_conv3').output,
        vgg.get_layer('block3_conv4').output,
        vgg.get_layer('block3_pool').output,
        vgg.get_layer('block4_conv1').output,
        vgg.get_layer('block4_conv2').output,
        vgg.get_layer('block4_conv3').output,
        vgg.get_layer('block4_conv4').output,
        vgg.get_layer('block4_pool').output,
        vgg.get_layer('block5_conv1').output,
        vgg.get_layer('block5_conv2').output,
        vgg.get_layer('block5_conv3').output,
        vgg.get_layer('block5_conv4').output,
        vgg.get_layer('block5_pool').output]

    model = Model(inputs=[vgg.input], outputs=vgg.outputs)
    model.trainable = False
    return model


def fitting_function(x, w):
    return (20 - x[:, 0]) / 12 * (w[0] + np.sum(x[:, 1:] * w[1:22], axis=1)) + (x[:, 0] - 8) / 12 * (w[22] + np.sum(
        x[:, 1:] * w[23:44], axis=1))


def logistic_function(x, w):
    return w[0] + (w[1] - w[0]) / ((w[2] + w[3] * np.exp(-x * w[4])) ** (1 / w[5]))


def robust_fitting_function(flatten_features, weights):
    logistic_parameters = weights[0:6]
    fitting_parameters = weights[6:]
    test_predictions = fitting_function(flatten_features, fitting_parameters)
    logistic_prediction = logistic_function(test_predictions, logistic_parameters)
    return logistic_prediction


def compute_vgg_features_prediction(gt, d, vgg_network):
    ground_truth_patches = np.copy(gt)
    distorted_patches = np.copy(d)
    vgg_features = np.zeros((ground_truth_patches.shape[0], 21))

    vgg_original_images = preprocess_input(ground_truth_patches)
    ground_truth_features = vgg_network(vgg_original_images)

    vgg_generated_images = preprocess_input(distorted_patches)
    prediction_features = vgg_network(vgg_generated_images)

    for i, (ground_truth_feature, prediction_feature) in enumerate(zip(ground_truth_features, prediction_features)):
        vgg_features[:, i] = tf.reduce_mean(tf.abs(tf.subtract(ground_truth_feature, prediction_feature)),
                                            axis=[1, 2, 3])

    return vgg_features


def get_prediction(features):
    parameters = np.loadtxt(f"pretrained/vgg_weights.csv", delimiter=",")
    predictions = robust_fitting_function(features, parameters)
    return predictions


def get_detection(real_image, foveated_image, vgg_network):
    splits = 5
    new_width = (real_image.shape[0] // 256) * 256
    delta_width = (real_image.shape[0] - new_width) // 2
    new_height = (real_image.shape[1] // 256) * 256
    delta_height = (real_image.shape[1] - new_height) // 2

    real_image = real_image[delta_width: new_width + delta_width, delta_height: new_height + delta_height, :]
    foveated_image = foveated_image[delta_width: new_width + delta_width, delta_height: new_height + delta_height, :]

    patches_vertical = new_height // 256
    patches_horizontal = new_width // 256

    foveated_patches = np.zeros((patches_horizontal * patches_vertical, 256, 256, 3))
    real_patches = np.zeros((patches_horizontal * patches_vertical, 256, 256, 3))
    center_distance = np.zeros((patches_horizontal * patches_vertical))

    counter = 0
    for i in range(patches_horizontal):
        for j in range(patches_vertical):
            foveated_patches[counter] = foveated_image[i * 256: i * 256 + 256, j * 256: j * 256 + 256, :]
            real_patches[counter] = real_image[i * 256: i * 256 + 256, j * 256: j * 256 + 256, :]
            pixels_distance = np.sqrt((new_width / 2 - (i + 0.5) * 256) ** 2 + (new_height / 2 - (j + 0.5) * 256) ** 2)
            center_distance[counter] = min(np.arctan(30 * pixels_distance / (1920 * 70)) * 180 / np.pi, 20)
            counter += 1

    total_patches = foveated_patches.shape[0]
    patches_per_split = total_patches // splits
    vgg_features = np.zeros((total_patches, 22))

    for i in range(splits):
        start = i * patches_per_split
        end = min((i + 1) * patches_per_split, total_patches)
        current_real_patches = real_patches[start: end]
        current_foveated_patches = foveated_patches[start: end]
        vgg_features[start: end, 0] = center_distance[start: end]
        vgg_features[start: end, 1:] = compute_vgg_features_prediction(current_real_patches, current_foveated_patches, vgg_network)

    vgg_features = vgg_features[vgg_features[:, 0] >= 8]
    detection = np.mean(get_prediction(vgg_features))
    return detection


def predict(args):
    vgg_network = make_vgg()
    real_image = np.asarray(Image.open(args.real)).astype(float)
    distorted_image = np.asarray(Image.open(args.distorted)).astype(float)
    detection = get_detection(real_image, distorted_image, vgg_network)
    print(f"Mean detection rate: {detection}")


def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--real", dest="real", help="path to the original image")
    parser.add_argument("-d", "--distorted", dest="distorted", help="path to the compared image")
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
