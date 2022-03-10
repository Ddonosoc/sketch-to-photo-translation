import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
stage = {3: 39, 4: 81, 5: 143}
base_model = ResNet50(input_shape=(224, 224, 3), weights="imagenet", include_top=True)
feature_model = Model(base_model.input, base_model.layers[stage[3]].output)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
