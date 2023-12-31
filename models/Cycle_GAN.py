import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

from models.mru_blocks import mru_block


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def CycleResnetGenerator(input_shape=(256, 256, 3),
                         output_channels=3,
                         dim=64,
                         n_downsamplings=2,
                         n_blocks=9,
                         norm='instance_norm',
                         use_mru=False,
                         apply_dropout=True):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        if apply_dropout:
            h = keras.layers.Dropout(0.5)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    print("Input")
    print(h.shape)
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    print("Enter stages")
    print(h.shape)
    # h = (None, 256, 256, 64)

    # 2
    K = None
    if not use_mru:
        for _ in range(n_downsamplings):
            dim *= 2
            h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
            h = Norm()(h)
            if apply_dropout:
                h = keras.layers.Dropout(0.5)(h)
            h = tf.nn.relu(h)
            print("Downsampling", dim)
            print(h.shape)
    else:
        out = h # input  h = (None, 256, 256, 64)
        I_i = h # input image  h = (None, 256, 256, 64)
        K = [h] # skip connection  h = (None, 256, 256, 64)
        for i in range(n_downsamplings):
            K_i = keras.layers.concatenate([out, I_i]) # [h, h] = (None, 256, 256, 128)
            out = mru_block(out, I_i, 32 * (2 ** i)) # out = (None, 128, 128, 32)
            if apply_dropout:
                out = keras.layers.Dropout(0.5)(out)
            I_i = keras.layers.AveragePooling2D(pool_size=2)(I_i)
            K.append(K_i)
            h = out

    # 3
    # h = (None, 64, 64, 256)
    for _ in range(n_blocks):
        h = _residual_block(h)
        print("Residual block", dim)
        print(h.shape)

    # 4
    if not use_mru:
        for _ in range(n_downsamplings):
            dim //= 2
            h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            print("Upsampling", dim)
            print(h.shape)
    else:
        for i in range(n_downsamplings):
            h = mru_block(h, K[n_downsamplings - i], 128 / (2 ** i) if i < n_downsamplings - 1 else 3, deconv=True)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)
    print("Output")
    print(h.shape)

    return keras.Model(inputs=inputs, outputs=h)


def CycleConvDiscriminator(input_shape=(256, 256, 3),
                           dim=64,
                           n_downsamplings=3,
                           norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    print("DISCRIMINATOR")
    h = inputs = keras.Input(shape=input_shape)
    print("Input")
    print(h.shape)
    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    print("Preprocess")
    print(h.shape)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        print("Downsampling", dim)
        print(h.shape)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    print("Process", dim)
    print(h.shape)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)
    print("Output")
    print(h.shape)

    return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                        1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
