import tensorflow.keras as keras
import models.Cycle_GAN as Cycle_GAN
lr = 0.0002
epochs = 10
len_dataset = 1000
epoch_decay = 100
beta_1 = 0.5


def get_optimizers(lr, epochs, len_dataset, epoch_decay, beta_1):
    """
    Get the optimizers for the CycleGAN model.
    """
    G_lr_scheduler = Cycle_GAN.LinearDecay(lr, epochs * len_dataset, epoch_decay * len_dataset)
    D_lr_scheduler = Cycle_GAN.LinearDecay(lr, epochs * len_dataset, epoch_decay * len_dataset)
    G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=beta_1)
    return G_optimizer, D_optimizer, G_lr_scheduler, D_lr_scheduler




