"""
Synthesize textures from Gram matrix of VGG features based on
Gatys et al., Texture Synthesis Using Convolutional Neural
Networks, 2015.
"""
import tensorflow as tf
from tensorflow.keras import Model  # , Input
import tensorflow_probability as tfp
import numpy as np
import os
import matplotlib.pyplot as plt
from vgg19 import VGG19
import datetime
import sys
import scipy.optimize as scio
import getopt
import math
import glob
from argparse import ArgumentParser


def read_image(filename):
    im = tf.io.read_file(filename)
    if filename.lower().endswith('png'):
        im = tf.image.decode_png(im)
    elif filename.lower().endswith('jpg') or filename.lower().endswith('jpeg'):
        im = tf.image.decode_jpeg(im)
    else:
        assert False, 'input image extension not supported'
    return im


class TextureSynthesizer:
    def __init__(self, output_dims=(224, 224)):
        self.output_dims = output_dims
        self.step_counter = 0
        self.lastGradient = None
        self.gram_target = None
        self.ml_target = None
        self.fs_target = None
        self.current_layer = -1
        self.current_iteration = 0

        self.vgg = VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        for n in range(len(self.vgg.layers)):
            self.vgg.layers[n].trainable = False

        vgglayer_names = ('block1_conv1',
                          'block1_conv2',
                          'block1_pool',
                          'block2_conv1',
                          'block2_conv2',
                          'block2_pool',
                          'block3_conv1',
                          'block3_conv2',
                          'block3_conv3',
                          'block3_conv4',
                          'block3_pool',
                          'block4_conv1',
                          'block4_conv2',
                          'block4_conv3',
                          'block4_conv4',
                          'block4_pool',
                          'block5_conv1',
                          'block5_conv2',
                          'block5_conv3',
                          'block5_conv4',
                          'block5_pool')

        self.averages = None
        npzfile = np.load('act_mean.npz', allow_pickle=True)
        averages = []
        for e in npzfile['arr_0']:
            averages.append(tf.reshape(e[0][0][0], (1, 1, 1, -1)))
        self.averages = []
        self.stdev = None
        npzfile = np.load('act_stdev.npz', allow_pickle=True)
        stdevs = []
        for e in npzfile['arr_0']:
            stdevs.append(tf.reshape(e[0][0][0], (1, 1, 1, -1)))
        self.stdev = []

        outputs = []
        for layer in self.vgg.layers:
            if layer.name in vgglayer_names:
                idx = vgglayer_names.index(layer.name)
                outputs.append(layer.output)
                if (self.averages is not None) and (self.stdev is not None):
                    self.averages.append(averages[idx])
                    self.stdev.append(stdevs[idx])
        self.loss_network = Model(inputs=self.vgg.input,
                                  outputs=outputs)

    def vectorize_and_normalize_features(self, im, included_layers):
        features = self.loss_network(im)
        features = [features[i] for i in included_layers]
        averages = [self.averages[i] for i in included_layers]
        stdev = [self.stdev[i] for i in included_layers]
        if type(features) is not list:
            features = [features]
        if (averages is not None) and (stdev is not None):
            for i in range(len(features)):
                features[i] = (features[i] - averages[i]) / (stdev[i] + 1e-22)
        else:
            raise ValueError("Cannot normalize feature maps. Mean and stdev not available.")
        features_vect = []
        for f in features:
            features_vect.append(tf.reshape(f, (f.shape[0], f.shape[1] * f.shape[2], f.shape[3])))
        return features_vect

    def gram_mat(self, im, included_layers):
        features_vect = self.vectorize_and_normalize_features(im, included_layers)
        result = []
        ml = []
        for f in features_vect:
            result.append(tf.einsum('aki,akj->aij', f, f))
            ml.append(f.shape[1] * f.shape[2])
        return result, ml

    def gram_loss(self, source, included_layers):
        gram_source, ml_source = self.gram_mat(source, included_layers)
        gram_target, ml_target = self.gram_target, self.ml_target
        layerloss = []
        n = 0
        for gs, gt in zip(gram_source, gram_target):
            c_s = 2 * gs.shape[1] * ml_source[n]
            c_t = 2 * gt.shape[1] * ml_target[n]
            layerloss.append(tf.reduce_sum(tf.square(gs / c_s - gt / c_t), axis=(1, 2)))
            n += 1
        return tf.reduce_mean(layerloss)

    def step(self, source, target, included_layers):
        with tf.GradientTape() as tape:
            tape.watch(source)
            if self.gram_target is None or self.ml_target is None:
                self.gram_target, self.ml_target = self.gram_mat(target, included_layers)
            if self.fs_target is None:
                self.fs_target = self.vectorize_and_normalize_features(target, included_layers)
            gl = self.gram_loss(source, included_layers)
            loss = gl
        gradient = tape.gradient(loss, source)
        self.lastGradient = gradient
        self.step_counter += 1
        return loss, gradient

    def synthesize(self, target, included_layers, mask):
        n_ch = target.shape[3]
        output_shape = target.shape
        output_dims = target.shape[1:3]

        source = tf.random.normal((1,) + output_dims + (n_ch,), mean=0.0, stddev=0.01)
        source = source * (1 - mask) + target * mask

        def val_fn(x):
            source_tensor = tf.convert_to_tensor(tf.reshape(x.astype(np.float32), output_shape))
            target_tensor = tf.convert_to_tensor(target)

            val = self.step(source_tensor, target_tensor, included_layers)[0]
            return val

        def grad_fn(x):
            if mask is None:
                masked_gradient = self.lastGradient
            else:
                masked_gradient = (1 - mask) * self.lastGradient
            grad = tf.reshape(masked_gradient, (-1,)).numpy().astype(np.float64)
            return grad

        self.gram_target, self.ml_target = None, None
        self.fs_target = None

        self.current_iteration = 0
        result = scio.minimize(val_fn, source, method='l-bfgs-b', jac=grad_fn,
                               options={'disp': False, 'ftol': 1e-12, 'gtol': 1e-14, 'maxcor': 10})

        niter = result.nit
        synthesized = np.reshape(result.x, source.shape)
        return synthesized, niter, mask


def write_im(im, filename):
    im_clip = tf.clip_by_value(im, -1.0, 1.0)
    tf.io.write_file(filename, tf.image.encode_png(tf.cast(im_clip[0] * 127.5 + 127.5, tf.uint8)))


def gaussian_kernel(std):
    distribution = tfp.distributions.Normal(0.0, std)
    size = tf.math.ceil(3 * std)
    vals = distribution.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def blur_image(image, std):
    kernel = gaussian_kernel(std)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    blurred_r = tf.nn.conv2d(image[:, :, :, 0, tf.newaxis], kernel, strides=[1, 1],
                             padding='SAME')[:, :, :, 0]
    blurred_g = tf.nn.conv2d(image[:, :, :, 1, tf.newaxis], kernel, strides=[1, 1],
                             padding='SAME')[:, :, :, 0]
    blurred_b = tf.nn.conv2d(image[:, :, :, 2, tf.newaxis], kernel, strides=[1, 1],
                             padding='SAME')[:, :, :, 0]
    blurred = tf.stack([blurred_r, blurred_g, blurred_b], axis=3)
    return blurred


def run_texture_synthesis(input_im, constraint_sampling, synthesizer):
    image_center = np.array([input_im.shape[:2]])
    image_center = image_center / 2

    input_im = tf.expand_dims(input_im, axis=0)
    x, y = tf.meshgrid(tf.range(input_im.shape[2]), tf.range(input_im.shape[1]))
    constant_sampling_rate = constraint_sampling

    distances = tf.sqrt(
        (tf.cast(y, tf.float32) - image_center[0][0]) ** 2 + (tf.cast(x, tf.float32) - image_center[0][1]) ** 2)
    sampling = tf.zeros((distances.shape)) + constant_sampling_rate
    probability = tf.random.uniform(sampling.shape, maxval=1.0)
    full_sampling_mask = tf.where(probability <= sampling, 1.0, 0.0)
    full_sampling_mask = full_sampling_mask[tf.newaxis, :, :, tf.newaxis]
    full_sampling_mask = tf.tile(full_sampling_mask, [1, 1, 1, 3])
    full_sampling_mask = blur_image(full_sampling_mask, 1.0)

    layers = [0, 1, 4, 8, 13, 18]
    synth_im, niter, sampling_pattern = synthesizer.synthesize(input_im, included_layers=layers,
                                                               mask=full_sampling_mask)
    return synth_im


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--sampling", dest="sampling",
                        help="guided sampling rate of the synthesis")
    parser.add_argument("-o", "--output", dest="output",
                        help="the folder for generated images")
    parser.add_argument("-i", "--input", dest="input",
                        help="the folder with input images")

    args = parser.parse_args()

    all_paths = glob.glob(f'{args.input}/*')
    synthesizer = TextureSynthesizer(output_dims=(256, 256))

    for path in all_paths:
        filename = path.split('\\')[-1]
        output_path = f'{args.output}/{filename}'
        input_im = read_image(path)
        input_im = tf.cast(input_im, tf.float32) / 127.5 - 1
        synth_im = run_texture_synthesis(input_im=input_im, constraint_sampling=float(args.sampling),
                                         synthesizer=synthesizer)
        write_im(synth_im, output_path)


if __name__ == '__main__':
    main()
