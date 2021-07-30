import tensorflow as tf
from tensorflow.keras import layers, Model


class DiscriminatorResidualBlock(layers.Layer):
    def __init__(self, n_filters, pooling_size, kernel_size):
        super(DiscriminatorResidualBlock, self).__init__()

        self.convolution_block = tf.keras.Sequential()
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))
        self.convolution_block.add(
            layers.Conv2D(n_filters, (kernel_size, kernel_size), 1, padding='same', kernel_initializer='glorot_normal'))
        self.convolution_block.add(layers.LeakyReLU(alpha=0.2))

        self.extend_input_layer = layers.Conv2D(n_filters, (1, 1), 1, padding='valid',
                                                kernel_initializer='glorot_normal')

        self.pooling_layer = layers.AveragePooling2D(padding='same', pool_size=pooling_size)

    def call(self, inputs, training=True):
        convoluted = self.convolution_block(inputs, training=training)
        extended = self.extend_input_layer(inputs, training=training)
        concatenated = convoluted + extended
        pooled = self.pooling_layer(concatenated, training=training)
        return pooled


class SpatialDiscriminator(Model):
    def __init__(self, pooling_size, filters, kernel_size):
        super(SpatialDiscriminator, self).__init__()

        self.model = tf.keras.Sequential()
        for i in range(len(filters)):
            self.model.add(DiscriminatorResidualBlock(filters[i], pooling_size, kernel_size))

    def call(self, inputs, training=True, **kwargs):
        output = self.model(inputs, training=training)
        return output


class Discriminator(Model):
    def __init__(self, pooling_size, filters, kernel_size, patch_size):
        super(Discriminator, self).__init__()

        self.spatial_discriminator = tf.keras.Sequential()
        self.spatial_discriminator.add(SpatialDiscriminator(pooling_size, filters, kernel_size))
        self.spatial_discriminator.add(layers.Flatten())
        self.spatial_discriminator.add(layers.Dense(1))
        self.patch_size = patch_size

    @tf.function
    def call(self, inputs, training=True):
        batch_size = inputs.shape[0]
        sliced_input = tf.image.extract_patches(images=inputs, sizes=[1, self.patch_size, self.patch_size, 1],
                                                strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1],
                                                padding='VALID')
        sliced_input = tf.reshape(sliced_input, [-1, self.patch_size, self.patch_size, 3])

        spatial_output = self.spatial_discriminator(sliced_input, training=training)
        spatial_output = tf.reshape(spatial_output, [batch_size, -1])
        spatial_output = tf.math.reduce_mean(spatial_output, 1)
        return spatial_output
