import tensorflow as tf
import argparse
import os
import datetime

from models.Cycle_LOSS import get_adversarial_losses_fn
from models.Pix2Pix_GAN import pix2pix_generator, pix2pix_discriminator
from models.Cycle_GAN import CycleResnetGenerator, CycleConvDiscriminator
from models.CycleUtils import get_optimizers
from models import Pix2PixUtils
from image_processing.image_process import load_image_train, load_image_test, load_image_trainv2
from config.configs import Config
from runner import fit, cycle_step, train_step
from PIL import Image


configs = Config("training_checkpoints_scribbler_purse")
configs.scribbler = True
generator = [pix2pix_generator(configs)]
discriminator = [pix2pix_discriminator()]
generator_optimizer = Pix2PixUtils.generator_optimizer
discriminator_optimizer = Pix2PixUtils.discriminator_optimizer
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

eval_dataset = tf.data.Dataset.list_files(configs.dataset_foldername + '*.png', shuffle=False)
AUTOTUNE = tf.data.experimental.AUTOTUNE
eval_dataset = eval_dataset.map(load_image_trainv2, num_parallel_calls=AUTOTUNE)

checkpoint_fname = configs.checkpoint_dir + '/ckpt-'
print("Evaluating...")
for i in range(0, 90):
    try:
        checkpoint.restore(f"{checkpoint_fname}{i}")
        for images in eval_dataset:
            filename = images[1].numpy().decode().split(configs.symbol_replacement)[-1]
            images = tf.expand_dims(images[0], 0)
            # generate_images_v2(generator, images)
            # print(images.shape)
            prediction = generator[0](images, training=True)
            # print(images[0].numpy())
            prediction = Image.fromarray(((prediction[0].numpy() * 0.5 + 0.5) * 255).astype('uint8'), 'RGB')
            # prediction = Image.fromarray((images[0].numpy()).astype('uint8'), 'RGB') # Forma color invertido
            prediction.save(f"{configs.folder_dest}{i}epoch{filename}")
            break
    except Exception as e:
        print(e)
