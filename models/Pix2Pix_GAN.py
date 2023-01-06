"""Implementation of Pix2Pix Neural Network"""
import math

import tensorflow as tf
import models.SwinTransformerBlock as SwinTransformerBlock


def downsample(filters, size, apply_batchnorm=True):
    """
    Downsample Layer for a GAN network
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Upsample Layer for a GAN network
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def pix2pix_generator(config):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(512, 4),  # (batch_size, 32, 32, 256)
        # downsample(512, 4),  # (batch_size, 16, 16, 512)
        # downsample(512, 4),  # (batch_size, 8, 8, 512)
        # downsample(512, 4),  # (batch_size, 4, 4, 512)
        # downsample(512, 4),  # (batch_size, 2, 2, 512)
        # downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs
    # Downsampling through the model
    skips = []
    i = 0
    for down in down_stack:
        if i == -1:
            x = SwinTransformerBlock.patch_embed(x, img_size=(256, 256))
            embed_dim = 48
            print(x.get_shape().as_list())
            x = SwinTransformerBlock.transformer(2, embed_dim * 2, max(embed_dim // 32, 4), x, 0.1)
            print(x.get_shape().as_list())
            x = tf.reshape(x, (-1, 64, 64, 96))
            x = upsample(64, 4)(x)
            i += 1
        else:
            x = down(x)

        skips.append(x)

    embed_dim = 256
    for t in range(4):
        size = x.get_shape().as_list()
        x = tf.reshape(x, (-1, size[1] * size[2], size[3]))
        x = SwinTransformerBlock.transformer(2, embed_dim * 2, max(embed_dim // 32, 4), x, 0.1)
        x = tf.reshape(x, (-1, size[1], size[2], size[3]))
        x = downsample(512, 4)(x)
        skips.append(x)
    x = downsample(512, 4)(x)
    x = downsample(512, 4)(x)
    # size = x.get_shape().as_list()
    # x = tf.reshape(x, (-1, size[1] * size[2], size[3]))
    # x = SwinTransformerBlock.transformer(2, embed_dim * 2, max(embed_dim // 32, 4), x, 0.1)
    # x = tf.reshape(x, (-1, size[1], size[2], size[3]))
    # x = downsample(512, 4)(x)
    # x = tf.keras.layers.Concatenate()([x, trans_skip])
    skips = reversed(skips)
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def pix2pix_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
