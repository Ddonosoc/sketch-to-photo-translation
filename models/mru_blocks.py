"""Implementation of Masked Residual Units (SketchyGAN)"""
import tensorflow as tf
import tensorflow.keras as keras

def mru_block(feature_maps, input_image, filter_depth, deconv=False):
    """
    Definition of a Masked Residual Unit Block according to SketchyGAN
    :param feature_maps: The feature map computed by the network
    :param input_image: The Input
    :param filter_depth: Filter of Convolutional layer
    :param deconv: Upsample (True) or Downsample (False)
    :return feature_map y_i
    """
    out_size = feature_maps.get_shape().as_list()[-1]

    if deconv:
        feature_maps = tf.keras.layers.Conv2DTranspose(filters=out_size, kernel_size=(2, 2), strides=(2, 2))(
            feature_maps)

    merge_one = tf.keras.layers.concatenate([feature_maps, input_image])

    # m_i
    conv_sig_m = tf.keras.layers.Conv2D(filters=out_size, kernel_size=(3, 3), padding="same", activation="sigmoid")(
        merge_one)

    # n_i
    conv_sig_n = tf.keras.layers.Conv2D(filters=filter_depth, kernel_size=(3, 3), padding="same", activation="sigmoid")(
        merge_one)

    mul_lower = tf.keras.layers.multiply([conv_sig_m, feature_maps])

    merge_two = tf.keras.layers.concatenate([mul_lower, input_image])

    conv_func_one = tf.keras.layers.Conv2D(filters=filter_depth, kernel_size=(3, 3), padding="same", activation="relu")(
        merge_two)
    print("CONV_FUNC_ONE")
    print(conv_func_one.shape)

    mul_higher = tf.keras.layers.multiply([conv_sig_n, conv_func_one])

    ni_subs = tf.keras.layers.Lambda(lambda x: 1 - x)(conv_sig_n)

    conv_feature_map = tf.keras.layers.Conv2D(filters=filter_depth, kernel_size=(3, 3), padding="same", activation="relu")(feature_maps)

    mul_final = tf.keras.layers.multiply([conv_feature_map, ni_subs])

    sum_final = tf.keras.layers.Add()([mul_higher, mul_final])

    if not deconv:
        sum_final = tf.keras.layers.MaxPooling2D(pool_size=2)(sum_final)

    # Extra
    sum_final = tf.keras.layers.Activation("relu")(sum_final)
    return sum_final


def get_encoder(I):
    K1 = keras.layers.concatenate([I, I])
    new = mru_block(I, I, 32)
    I2 = keras.layers.AveragePooling2D(pool_size=2)(I)

    K2 = keras.layers.concatenate([new, I2])
    new = mru_block(new, I2, 64)
    I3 = keras.layers.AveragePooling2D(pool_size=2)(I2)

    K3 = keras.layers.concatenate([new, I3])
    new = mru_block(new, I3, 128)
    I4 = keras.layers.AveragePooling2D(pool_size=2)(I3)

    K4 = keras.layers.concatenate([new, I4])
    out = mru_block(new, I4, 256)
    # I5 = keras.layers.AveragePooling2D(pool_size=2)(I4)
    #
    # K5 = keras.layers.concatenate([out, I5])

    encoder = keras.Model(I, out, name='encoder')

    return encoder, K4, K3, K2, K1


def get_decoder(feature_map, K4, K3, K2, K1, I):
    decoder_in = keras.layers.Input(shape=(int(feature_map.shape[1]), int(feature_map.shape[2]), int(feature_map.shape[3])))

    new = mru_block(decoder_in, K4, 128, deconv=True)
    new = mru_block(new, K3, 64, deconv=True)
    new = mru_block(new, K2, 32, deconv=True)
    out = mru_block(new, K1, 3, deconv=True)

    decoder = keras.Model([decoder_in, I], out, name='decoder')

    return decoder


def get_generator(img_shape):
    I = keras.layers.Input(shape=img_shape)

    encoder, K4, K3, K2, K1 = get_encoder(I)

    decoder = get_decoder(encoder.output, K4, K3, K2, K1, I)

    encoded = encoder(I)

    out = decoder([encoded, I])

    return keras.Model([I], out, name='generator_model')


def get_discriminator(img_shape):
    I = keras.layers.Input(shape=img_shape)
    out = mru_block(I, I, 8)

    return keras.Model(I, out, name='discriminator_skeleton')