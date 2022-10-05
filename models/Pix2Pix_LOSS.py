import tensorflow as tf
from models import Pix2PixUtils
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


def generator_loss(disc_generated_output, gen_output, target, config):
    gan_loss = Pix2PixUtils.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (config.LAMBDA * l1_loss)

    if config.scribbler:
        gen_im = tf.image.resize(gen_output[0], [224, 224])
        tv_loss = tf.image.total_variation(gen_im)
        gen_im = tf.expand_dims(gen_im, 0)
        gen_im = preprocess_input(gen_im)

        tar_im = tf.image.resize(target[0], [224, 224])
        tar_im = tf.expand_dims(tar_im, 0)
        tar_im = preprocess_input(tar_im)
        feature_loss = tf.norm(Pix2PixUtils.feature_model(gen_im) - Pix2PixUtils.feature_model(tar_im))
        total_gen_loss += config.TV_WEIGHT * tv_loss
        total_gen_loss += config.F_WEIGHT * feature_loss
        return [total_gen_loss, gan_loss, l1_loss, tv_loss], ["gen_total_loss", "gan_loss", "l1_loss",
                                                                            "tv_loss"]
        # return [total_gen_loss, gan_loss, l1_loss, tv_loss, feature_loss], ["gen_total_loss", "gan_loss", "l1_loss", "tv_loss", "feature_loss"]

    return [total_gen_loss, gan_loss, l1_loss], ["gen_total_loss", "gan_loss", "l1_loss"]


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = Pix2PixUtils.loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = Pix2PixUtils.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
