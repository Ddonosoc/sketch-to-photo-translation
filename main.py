import tensorflow as tf
import argparse
import os
import datetime
from models.Pix2Pix_GAN import pix2pix_generator, pix2pix_discriminator
from models import Pix2PixUtils
from image_processing.image_process import load_image_train, load_image_test, load_image_trainv2
from config.configs import Config
from runner import fit
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Quimera Model")
    parser.add_argument("-mode", type=str, choices=["train", "test"], help="train or test", required=False,
                        default="train")
    parser.add_argument("-network", type=str, choices=["simple", "mru", "cycle", "scribbler"],
                        help="Use generator and discriminator as example models like simple pix2pix, with mru, with cycle consistency or simple scribbler",
                        required=False, default="simple")

    parser.add_argument("-checkpoint", type=str, help="Checkpoint folder name", required=False, default="training_checkpoints")

    pargs = parser.parse_args()
    configs = Config(pargs.checkpoint)
    if pargs.network == "simple":
        generator = pix2pix_generator(configs)
        discriminator = pix2pix_discriminator()
        generator_optimizer = Pix2PixUtils.generator_optimizer
        discriminator_optimizer = Pix2PixUtils.discriminator_optimizer

    elif pargs.network == "scribbler":
        configs.scribbler = True
        generator = pix2pix_generator(configs)
        discriminator = pix2pix_discriminator()
        generator_optimizer = Pix2PixUtils.generator_optimizer
        discriminator_optimizer = Pix2PixUtils.discriminator_optimizer
    else:
        raise NotImplemented

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(configs.checkpoint_dir))

    print(len(os.listdir(configs.folder_dataset_train)))

    print(len(os.listdir(configs.folder_dataset_test)))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.list_files(configs.folder_dataset_train + '*.png')
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.shuffle(configs.BUFFER_SIZE)
    train_dataset = train_dataset.batch(configs.BATCH_SIZE)

    try:
        test_dataset = tf.data.Dataset.list_files(configs.folder_dataset_test + '*.png', shuffle=False)
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(configs.folder_dataset_test + '*.png', shuffle=False)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(configs.BATCH_SIZE)

    log_dir = configs.folder + "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if pargs.mode == "train":
        fit(train_dataset, test_dataset, steps=80000, checkpoint=checkpoint,
            generator=generator, discriminator=discriminator, config=configs, summary_writer=summary_writer)
    else:
        eval_dataset = tf.data.Dataset.list_files(configs.dataset_foldername + '*.png', shuffle=False)
        eval_dataset = eval_dataset.map(load_image_trainv2, num_parallel_calls=AUTOTUNE)
        for images in eval_dataset:
            filename = images[1].numpy().decode().split(configs.symbol_replacement)[-1]
            print(filename)
            images = tf.expand_dims(images[0], 0)
            # generate_images_v2(generator, images)
            # print(images.shape)
            prediction = generator(images, training=True)
            # print(images[0].numpy())
            prediction = Image.fromarray(((prediction[0].numpy() * 0.5 + 0.5) * 255).astype('uint8'), 'RGB')
            # prediction = Image.fromarray((images[0].numpy()).astype('uint8'), 'RGB') # Forma color invertido
            prediction.save(f"{configs.folder_dest}{filename}")
