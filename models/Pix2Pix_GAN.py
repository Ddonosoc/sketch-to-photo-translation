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


def upsample(filters, size, apply_dropout=False, strides=2, output_padding=None):
    """
    Upsample Layer for a GAN network
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        output_padding=output_padding,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def identity_block(x, filter):
    x_skip = x

    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    x_skip = x

    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


# def pix2pix_generator(config):
#     inputs = tf.keras.layers.Input(shape=[256, 256, 3])
#     up_stack = [
#         upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
#         upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#         upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
#         upsample(64, 4),  # (batch_size, 16, 16, 1024)
#         upsample(64, 4)
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4,
#                                            strides=2,
#                                            padding='same',
#                                            kernel_initializer=initializer,
#                                            activation='tanh')  # (batch_size, 256, 256, 3)

#     x = inputs
#     # Downsampling through the model
#     skips = []
#     # x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
#     skips.append(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
#     block_layers = [3, 4, 6, 3]
#     filter_size = 64
#     for i in range(4):
#         if i == 0:
#             for j in range(block_layers[i]):
#                 x = identity_block(x, filter_size)
#         else:
#             # One Residual/Convolutional Block followed by Identity blocks
#             # The filter size will go on increasing by a factor of 2
#             filter_size = filter_size * 2
#             x = convolutional_block(x, filter_size)
#             for j in range(block_layers[i] - 1):
#                 x = identity_block(x, filter_size)
#         print(x.get_shape().as_list())


#         skips.append(x)

#     skips = reversed(skips[:-1])
#     # Upsampling and establishing the skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = tf.keras.layers.Concatenate()([x, skip])

#     x = last(x)
#     return tf.keras.Model(inputs=inputs, outputs=x)


# def pix2pix_generator_color(config):
#     inputs = tf.keras.layers.Input(shape=[256, 256, 3])
#     colors = tf.keras.layers.Input(shape=[96])

#     up_stack = [
#         upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
#         upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#         upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
#         upsample(64, 4),  # (batch_size, 16, 16, 1024)
#         upsample(64, 4)   # (batch_size, 8, 8, 512)
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4,
#                                            strides=2,
#                                            padding='same',
#                                            kernel_initializer=initializer,
#                                            activation='tanh')  # (batch_size, 256, 256, 3)

#     x = inputs
#     # Downsampling through the model
#     skips = []
#     # x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
#     skips.append(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
#     block_layers = [3, 4, 6, 3]
#     filter_size = 64
#     for i in range(4):
#         if i == 0:
#             for j in range(block_layers[i]):
#                 x = identity_block(x, filter_size)
#         else:
#             # One Residual/Convolutional Block followed by Identity blocks
#             # The filter size will go on increasing by a factor of 2
#             filter_size = filter_size * 2
#             x = convolutional_block(x, filter_size)
#             for j in range(block_layers[i] - 1):
#                 x = identity_block(x, filter_size)

#         skips.append(x)

#     skips = reversed(skips[:-1])
#     # Upsampling and establishing the skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         print(x.get_shape().as_list(), skip.get_shape().as_list())
#         x = tf.keras.layers.Concatenate()([x, skip])

#     x = last(x)
#     return tf.keras.Model(inputs=inputs, outputs=x)


# def pix2pix_discriminator():
#     initializer = tf.random_normal_initializer(0., 0.02)

#     inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
#     tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

#     x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

#     down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
#     down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
#     down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

#     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
#     conv = tf.keras.layers.Conv2D(512, 4, strides=1,
#                                   kernel_initializer=initializer,
#                                   use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

#     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

#     leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

#     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

#     last = tf.keras.layers.Conv2D(1, 4, strides=1,
#                                   kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

#     return tf.keras.Model(inputs=[inp, tar], outputs=last)



def pix2pix_generator(config):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
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
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def convrelu(in_size, out_size, kernel_size=1, activation=tf.keras.layers.ReLU(), space_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(out_size, kernel_size=kernel_size, padding='same', kernel_initializer=initializer))
    result.add(activation)
    return result



def global_network(img_size, x,dim, n_up=0, add_L=1):
    if add_L:
        oneD = convrelu(16, 128)
    else:
        oneD = convrelu(11, 128)

    print(x.shape)
    twoD = convrelu(128, 256)
    threeD = convrelu(256, 512)
    fourD = convrelu(512, 512)
    out = oneD(x)
    if dim >= 256:
        n = 4
        out = twoD(out)
    if dim == 512:
        n = 8
        out = threeD(out)
        out = fourD(out)
    for i in range(n_up):
        out = upsample(dim, 4)(out)
    print(out.shape)
    return out




def pix2pix_generator_color_palette(config):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    colors_input = tf.keras.layers.Input(shape=[96])
    out_color = tf.reshape(colors_input, (-1, 4, 4, 6))
    out_color = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(out_color)
    out_color_128 = upsample(128, 4)(out_color) #128x128
    out_color_64 = downsample(256, 4)(out_color_128) #64x64
    out_color = downsample(512, 4)(out_color_64) #32x32
    out_color_512 = downsample(512, 4)(out_color) #16x16
    print("OUT")
    print(out_color.shape)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        # downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
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
    for i, down in enumerate(down_stack):
        x = down(x)
        if i == 3:
            x = x + out_color_512
        skips.append(x)

    skips = list(reversed(skips[:-1]))

    i = 0
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        i += 1
        if i == len(skips) - 1:
            x = x + out_color_64

    x = x + out_color_128
    x = last(x)

    return tf.keras.Model(inputs=[inputs, colors_input], outputs=x)





def pix2pix_generator_color(config):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    colors = tf.keras.layers.Input(shape=[96])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
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

    c_y = tf.reshape(colors, [-1, 4, 4, 6])
    c_y = downsample(512, 4)(c_y) # (batch_size, 2, 2, 512)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips)

    x = tf.keras.layers.Concatenate()([x, c_y])
    x = downsample(512, 4)(x) # (batch_size, 1, 1, 512)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs, colors], outputs=x)


def pix2pix_discriminator_color():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    colors = tf.keras.layers.Input(shape=[96])
    inp = tf.keras.layers.concatenate([inp, colors])

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
