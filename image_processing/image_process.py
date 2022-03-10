import tensorflow as tf
from config.configs import IMG_HEIGHT, IMG_WIDTH

def load(image_file):
    """
    Read and decode an image file to a uint8 tensor
    """

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    """
    Resize both input image and real image to heightxwidth, return modified images
    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, height, width):
    """
    Crops randomly both input and real images, accordingly to Pix2Pix
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    """
    Normalize both input and real images
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, height, width):
    """
    Apply several transformations to both input and real images, resizing to heightxwidth
    """
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image, height, width)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    """
    Load train Dataset Image and apply transformations to the input and real image. It must apply random jitter,
    because the Pix2Pixflow
    """
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    """
    Load test Dataset image. The Function doesn't apply transformation as random jitter, because those kind of
    transformation are applied during training phase
    """
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def loadv2(image_file):
    """
    Load Dataset Image and apply transformations to the input for evaluation.
    """
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image

    input_image = tf.cast(image, tf.float32)

    input_image = (input_image / 127.5) - 1

    return input_image


def load_image_trainv2(image_file):
    """
    Load Dataset image for evaluation
    """
    input_image = loadv2(image_file)

    return input_image, image_file

