import tensorflow as tf
import time
from models.Pix2Pix_LOSS import discriminator_loss, generator_loss
from models.Pix2PixUtils import generator_optimizer, discriminator_optimizer
from models.mru_blocks import mru_block


@tf.function
def train_step(input_image, target, step, generator, discriminator, config, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_losses, names = generator_loss(disc_generated_output,
                                    gen_output, target,
                                    config)
        # gen_total_loss, gen_gan_loss, gen_l1_loss, scribbler_loss, feature_loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_losses[0],
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        for loss, name in zip(gen_losses, names):
            tf.summary.scalar(name, loss, step=step//1000)


def fit(train_ds, test_ds, steps, checkpoint, generator, discriminator, config, summary_writer):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()

            # generate_images(generator, example_input, example_target)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step, generator, discriminator, config, summary_writer)
        # mru = mru_block(input_image, input_image, 64)
        # I2 = tf.keras.layers.AveragePooling2D(pool_size=2)(input_image)
        # mru_2 = mru_block(mru, I2, 128)
        #
        # print("MRU BLOCK INPUT")
        # print(mru_2.get_shape())
        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=config.checkpoint_prefix)
