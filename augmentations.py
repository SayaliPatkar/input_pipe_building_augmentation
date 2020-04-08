import tensorflow as tf
import numpy as np
import cv2
from numpy import random
from PIL import Image


def flip_horz(img: tf.Tensor) -> tf.Tensor:
    """Vertical flip augmentation
    Args:
        img: Image tensor to flip

    Returns:
        Augmented image tensor
    """
    # use tf.image.random_flip_left_right for randomnaess
    img = tf.image.flip_left_right(img)
    return img


def flip_vert(img: tf.Tensor) -> tf.Tensor:
    """Horizontal flip augmentation
    Args:
        img: Image tensor to flip

    Returns:
        Augmented image tensor
    """
    # use tf.image.flip_up_down for randomnaess
    img = tf.image.flip_up_down(img)
    return img

def rotate(img: tf.Tensor, max_angle) -> tf.Tensor:
    """Rotating images with max_angle
    can be easily achieved by tfa.image.rotate for tensorflow 2.0
    However it is very irksome downloading tensorflow-addon version 0.6.0 for
    windows user so this is the workaround

    Args:
        img: Image tensor to flip
        max_angle: max rotation angle in degrees

    Returns:
        Augmented image tensor
    """
    random.seed(1)
    angle = random.randint(-max_angle, max_angle, 1)
    rotated_image = tf.compat.v1.numpy_function(rotate_with_CV2, (img, angle) , tf.float32)
    return rotated_image

def rotate_with_CV2(image:np.array, angle):
    """Rotating images with max_angle using open_cv functions
    Args:
        img: Image array to rotate
        angle: rotation angle in degrees

    Returns:
        Augmented image array
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def rotate_with_CV2(image:np.array):
    """Rotating images with max_angle using open_cv functions
    Args:
        img: Image array to rotate
        angle: rotation angle in degrees

    Returns:
        Augmented image array
    """
    random.seed(1)
    angle = random.randint(-5, 5, 1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotated_tensor = tf.convert_to_tensor(rotated, dtype=tf.float32)
    return rotated_tensor

def scale(img: tf.Tensor, max_scaling) -> tf.Tensor:
    """Zooming/scaling image by random factor augmentation

    Args:
        img: Image tensor to rotate
        max_scaling: max permittable scaling factor

    Returns:
        Augmented image tensor
    """
    input_shape = tf.cast(tf.shape(img)[0:2], tf.int32)
    # Generate crop choices
    scales = list(np.arange(1.0-max_scaling, 1.0, 0.05))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    # crop and resize to produce results for all scales
    scaled_imgs = tf.image.crop_and_resize([img], boxes=boxes,
                        box_indices=np.zeros(len(scales)), crop_size=input_shape)
    # randomly return single scaled image
    return scaled_imgs[tf.random.uniform(shape=[], minval=0, maxval=len(scales),
                            dtype=tf.int32)]

def color(img: tf.Tensor) -> tf.Tensor:
    """Perform color augmentations adjusting hue, saturation, brightness
    and contrast
    Args:
        img: Image tensor

    Returns:
        Augmented image tensor
    """
    # max_delta [0, 0.5].
    img = tf.image.random_hue(img, max_delta= 0.1)

    # max_delta float, must be non-negative.
    img = tf.image.random_brightness(img, 0.05)

    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img

def gaussian_noise(img: tf.Tensor) -> tf.Tensor:
    """Adding Gaussian noise to the image
    Args:
        img: Image tensor

    Returns:
        Augmented image tensor
    """
    # Adding Gaussian noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32)
    noise_img = tf.add(img, noise)
    return noise_img
