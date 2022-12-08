import functools
import tensorflow as tf
import time
from models.Pix2Pix_LOSS import discriminator_loss, generator_loss
from models.Pix2PixUtils import generator_optimizer, discriminator_optimizer
from models.Cycle_LOSS import gradient_penalty


@tf.function
def train_step(input_image, target, step, generator, discriminator, config, summary_writer, **kwargs):
    generator = generator[0]
    discriminator = discriminator[0]
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
            tf.summary.scalar(name, loss, step=step)


@tf.function
def train_G(A, B, generator_data, discriminator_data, g_loss_fn, cycle_loss_fn, identity_loss_fn, config,
            generator_optimizer):
    G_A2B, G_B2A = generator_data
    D_A, D_B = discriminator_data

    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)
        pixpix = config.pix2pix
        if pixpix:
            A2B_d_logits = D_B([A2B, B], training=True)
            B2A_d_logits = D_A([B2A, A], training=True)
        else:
            A2B_d_logits = D_B(A2B, training=True)
            B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)

        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * config.cycle_loss_weight + (
                A2A_id_loss + B2B_id_loss) * config.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    generator_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(A, B, A2B, B2A, discriminator_data, d_loss_fn, config, d_optimizer):
    D_A, D_B = discriminator_data
    with tf.GradientTape() as t:
        if config.pix2pix:
            A_d_logits = D_A([A, A], training=True)
            B2A_d_logits = D_A([B2A, A], training=True)
            B_d_logits = D_B([B, B], training=True)
            A2B_d_logits = D_B([A2B, B], training=True)
        else:
            A_d_logits = D_A(A, training=True)
            B2A_d_logits = D_A(B2A, training=True)
            B_d_logits = D_B(B, training=True)
            A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=config.gradient_penalty_mode)
        D_B_gp = gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=config.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * config.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    d_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def cycle_step(input_image, target, step, generator, discriminator, config, summary_writer, loss_param,
               d_optimizer):
    cycle_loss_fn = loss_param['cycle_loss_fn']
    identity_loss_fn = loss_param['identity_loss_fn']
    g_loss_fn = loss_param['g_fn_loss']
    d_loss_fn = loss_param['d_fn_loss']

    A2B, B2A, G_loss_dict = train_G(A=input_image, B=target, generator_data=generator, discriminator_data=discriminator,
                                    g_loss_fn=g_loss_fn, cycle_loss_fn=cycle_loss_fn, identity_loss_fn=identity_loss_fn,
                                    config=config, generator_optimizer=generator_optimizer)

    D_loss_dict = train_D(A=input_image, B=target, A2B=A2B, B2A=B2A, discriminator_data=discriminator, config=config,
                          d_optimizer=d_optimizer, d_loss_fn=d_loss_fn)

    with summary_writer.as_default():
        for key in G_loss_dict:
            tf.summary.scalar(key, G_loss_dict[key], step=step)
        for key_loss in D_loss_dict:
            tf.summary.scalar(key_loss, D_loss_dict[key_loss], step=step)


def fit(train_ds, test_ds, steps, checkpoint, generator, discriminator, config, summary_writer,
        step_trainer=train_step, d_optimizer=None, loss_param=None):
    start = time.time()

    for epoch in range(10):
        print(f"Starting epoch {epoch}")
        for step, (input_image, target) in enumerate(train_ds):
            if (step) % 100 == 0:

                if step != 0:
                    print(f'Time taken for 100 steps: {time.time() - start:.2f} sec\n')

                start = time.time()

                # generate_images(generator, example_input, example_target)
                print(f"Step: {step // 1000}k")

            step_trainer(input_image=input_image, target=target, step=step, generator=generator,
                         discriminator=discriminator, config=config, summary_writer=summary_writer, d_optimizer=d_optimizer,
                         loss_param=loss_param)
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
            if (step + 1) % 300 == 0:
                checkpoint.save(file_prefix=config.checkpoint_prefix)
    checkpoint.save(file_prefix=config.checkpoint_prefix)
