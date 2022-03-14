"""Implementation of Masked Residual Units (SketchyGAN)"""
import tensorflow as tf


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
