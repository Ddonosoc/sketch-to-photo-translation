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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Quimera Model")
    parser.add_argument("-mode", type=str, choices=["train", "test"], help="train or test", required=False,
                        default="train")
    parser.add_argument("-network", type=str, choices=["simple", "mru", "scribbler", "cycle"],
                        help="Use generator and discriminator as example models like simple pix2pix, with mru, with cycle consistency or simple scribbler",
                        required=False, default="simple")

    parser.add_argument("-runner", type=str, choices=["simple", "cycle"], default="simple", required=False)

    parser.add_argument("-checkpoint", type=str, help="Checkpoint folder name", required=False,
                        default="training_checkpoints")

    parser.add_argument("-epochs", type=int, help="Number of epochs", required=False, default=10)

    pargs = parser.parse_args()
    configs = Config(pargs.checkpoint)

    len_dataset = configs.len_dataset

    if pargs.network != "cycle" and pargs.runner == "cycle":
        raise NotImplemented

    step_trainer = train_step
    losses_param = None
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

    elif pargs.network == "cycle":
        G_A2B = CycleResnetGenerator((configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))
        G_B2A = CycleResnetGenerator((configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))
        D_A2B = CycleConvDiscriminator((configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))
        D_B2A = CycleConvDiscriminator((configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))
        generator = G_A2B, G_B2A
        discriminator = D_A2B, D_B2A
        generator_optimizer, discriminator_optimizer, g_lr_scheduler, d_lr_scheduler = get_optimizers(configs.lr,
                                                                                                      configs.epochs,
                                                                                                      len_dataset,
                                                                                                      configs.epoch_decay,
                                                                                                      configs.beta_1)

        d_fn_loss, g_fn_loss = get_adversarial_losses_fn(configs.loss_mode)
        cycle_loss_fn = tf.losses.MeanAbsoluteError()
        identity_loss_fn = tf.losses.MeanAbsoluteError()

        losses_param = {
            "cycle_loss_fn": cycle_loss_fn,
            "identity_loss_fn": identity_loss_fn,
            "d_fn_loss": d_fn_loss,
            "g_fn_loss": g_fn_loss
        }

        step_trainer = cycle_step

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         G_A2B=G_A2B,
                                         G_B2A=G_B2A,
                                         D_A2B=D_A2B,
                                         D_B2A=D_B2A)
    else:
        raise NotImplemented

    if pargs.network != "cycle":
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
            generator=generator, discriminator=discriminator, config=configs,
            summary_writer=summary_writer, step_trainer=step_trainer, d_optimizer=discriminator_optimizer,
            loss_param=losses_param)
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
