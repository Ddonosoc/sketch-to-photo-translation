"""Implementation of Self-Attention Module"""
import tensorflow as tf
import tensorflow.keras as keras


def self_attention_block(feature_map, dim):
    """
    Definition of a self-attention module according to SAGAN
    :param feature_map: The feature maps computed by the network as input of this layer
    :return: The feature maps after applying the self-attention module
    """
    w, h, c = feature_map.shape.as_list()
    query = keras.layers.Conv2D(dim, (1, 1), padding='same')(feature_map)
    key = keras.layers.Conv2D(dim, (1, 1), padding='same')(feature_map)
    value = keras.layers.Conv2D(dim, (1, 1), padding='same')(feature_map)





    return True