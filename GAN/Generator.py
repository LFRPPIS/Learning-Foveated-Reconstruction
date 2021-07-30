import tensorflow as tf
from tensorflow.keras import layers, Model


class GeneratorResidualBlock(layers.Layer):
    def __init__(self, n_filters, pooling_size, kernel_size):
        super(GeneratorResidualBlock, self).__init__()

        self.convolution_block = tf.keras.Sequential()
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))

        self.extend_input_layer = tf.keras.Sequential()
        self.extend_input_layer.add(
            layers.Conv2D(n_filters, (1, 1), 1, padding='valid', kernel_initializer='glorot_normal'))
        self.extend_input_layer.add(layers.LeakyReLU(alpha=0.2))

        self.pooling_layer = layers.AveragePooling2D(padding='same', pool_size=pooling_size)

    def call(self, inputs, training=True):
        convoluted = self.convolution_block(inputs, training=training)
        extended = self.extend_input_layer(inputs, training=training)
        added = convoluted + extended
        pooled = self.pooling_layer(added, training=training)
        return pooled


class BottleneckBlock(layers.Layer):
    def __init__(self, n_filters, pooling_size, kernel_size):
        super(BottleneckBlock, self).__init__()

        self.convolution_block = tf.keras.Sequential()
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))

        self.extend_input_layer = tf.keras.Sequential()
        self.extend_input_layer.add(
            layers.Conv2D(n_filters, (1, 1), 1, padding='valid', kernel_initializer='glorot_normal'))
        self.extend_input_layer.add(layers.LeakyReLU(alpha=0.2))

        self.pooling_and_upsampling_block = tf.keras.Sequential()
        self.pooling_and_upsampling_block.add(layers.AveragePooling2D(padding='same', pool_size=pooling_size))
        self.pooling_and_upsampling_block.add(
            layers.UpSampling2D(interpolation='bilinear', size=(pooling_size, pooling_size)))

    def call(self, inputs, training=True):
        convoluted = self.convolution_block(inputs, training=training)
        extended = self.extend_input_layer(inputs, training=training)
        added = convoluted + extended
        pooled_and_upsampled = self.pooling_and_upsampling_block(added, training=training)
        return pooled_and_upsampled


class TemporalBlock(layers.Layer):
    def __init__(self, n_filters, pooling_size, kernel_size):
        super(TemporalBlock, self).__init__()

        self.convolution_block = tf.keras.Sequential()
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))

        self.extend_input_layer = tf.keras.Sequential()
        self.extend_input_layer.add(
            layers.Conv2D(n_filters, (1, 1), 1, padding='valid', kernel_initializer='glorot_normal'))
        self.extend_input_layer.add(layers.LeakyReLU(alpha=0.2))

        self.upsample_layer = layers.UpSampling2D(interpolation='bilinear', size=(pooling_size, pooling_size))

    def call(self, inputs, training=True):
        upsampled = self.upsample_layer(inputs, training=training)
        convoluted = self.convolution_block(upsampled, training=training)
        extended = self.extend_input_layer(upsampled, training=training)
        added = convoluted + extended
        return added


class Generator(Model):
    n_blocks = 0

    def __init__(self, pooling_size, filters, kernel_size):
        super(Generator, self).__init__()
        self.n_blocks = len(filters) - 1

        self.residual_block = [None] * self.n_blocks
        self.temporal_block = [None] * self.n_blocks

        for i in range(self.n_blocks):
            self.residual_block[i] = GeneratorResidualBlock(filters[i], pooling_size, kernel_size)

        self.bottleneck_block = BottleneckBlock(filters[-1], pooling_size, kernel_size)
        for i in range(self.n_blocks):
            self.temporal_block[i] = TemporalBlock(filters[-i - 2], pooling_size, kernel_size)

        self.convolution_to_image = tf.keras.Sequential()
        self.convolution_to_image.add(layers.Conv2D(3, (1, 1), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_to_image.add(layers.LeakyReLU(alpha=0.2))
        self.convolution_to_image.add(layers.Activation('tanh'))

    @tf.function
    def call(self, inputs, training=True):
        residual_block_output = [None] * self.n_blocks
        block_input = inputs
        for i in range(self.n_blocks):
            residual_block_output[i] = self.residual_block[i](block_input, training=training)
            block_input = residual_block_output[i]

        bottleneck_block_output = self.bottleneck_block(block_input, training=training)
        output = bottleneck_block_output

        for i in range(self.n_blocks):
            padding_vert = output.shape[1] - residual_block_output[-i - 1].shape[1]
            padding_hor = output.shape[2] - residual_block_output[-i - 1].shape[2]
            output = layers.concatenate([output[:, padding_vert:, padding_hor:, :], residual_block_output[-i - 1]],
                                        axis=3)
            output = self.temporal_block[i](output, training=training)

        padding_vert = output.shape[1] - inputs.shape[1]
        padding_hor = output.shape[2] - inputs.shape[2]
        output = self.convolution_to_image(output[:, padding_vert:, padding_hor:, :], training=training)
        return output
