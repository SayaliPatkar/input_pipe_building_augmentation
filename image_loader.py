import os
import cv2
import errno
import yamlargparse
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from glob import glob

import constants as const
import test as test_code
import augmentations as aug


def prepare_input_pipeline(config):
    mode = config.data_prep.mode
    channels = config.data_prep.channels
    resize = config.data_prep.resize
    if mode == 'augmentation_demo':
        demo_list = get_image_list(mode, config.input.train_path)
        aug_tensor = []
        for file in demo_list:
            im = load_image(file, resize, channels)
            if config.data_prep.augmentation:
                if config.augmentation_ops.max_rotation:
                    max_angle = config.augmentation_ops.max_rotation
                    rotated_ds = aug.rotate_with_CV2(im.numpy())
                if config.augmentation_ops.flip_vert:
                    flip_v_ds = aug.flip_vert(im)
                if config.augmentation_ops.flip_horz:
                    flip_h_ds = aug.flip_horz(im)
                if config.augmentation_ops.max_scaling:
                    max_scale = config.augmentation_ops.max_scaling
                    scaled_ds = aug.scale(im, max_scale)
                if config.augmentation_ops.color:
                    color_ds = aug.color(im)
                if config.augmentation_ops.gaussian_noise:
                    noise_ds = aug.gaussian_noise(im)
            arr = tf.stack([im, flip_v_ds, flip_h_ds, scaled_ds, color_ds, noise_ds, rotated_ds], 0)
            aug_tensor.append(arr)
        return aug_tensor
    elif mode == 'train':
        train_list = get_image_list(mode, config.input.train_path)
        list_ds = tf.data.Dataset.list_files(train_list)
        num_threads = 5
        train_ds = list_ds.map(lambda x: load_image(x, resize, channels), num_parallel_calls=num_threads)
        if config.data_prep.augmentation:
            if config.augmentation_ops.flip_vert:
                train_ds = train_ds.map(aug.flip_vert, num_parallel_calls=num_threads)
            if config.augmentation_ops.flip_horz:
                train_ds = train_ds.map(aug.flip_horz, num_parallel_calls=num_threads)
            if config.augmentation_ops.max_scaling:
                max_scale = config.augmentation_ops.max_scaling
                train_ds = train_ds.map(lambda x: aug.scale(x, max_scale),
                                             num_parallel_calls=num_threads)
            if config.augmentation_ops.color:
                train_ds = train_ds.map(aug.color, num_parallel_calls=num_threads)
            if config.augmentation_ops.gaussian_noise:
                train_ds = train_ds.map(aug.gaussian_noise, num_parallel_calls=num_threads)
            if config.augmentation_ops.max_rotation:
                max_angle = config.augmentation_ops.max_rotation
                train_ds = train_ds.map(lambda x: aug.rotate(x, max_angle),
                                          num_parallel_calls=num_threads)
            return train_ds
    test_list = get_image_list(mode, config.input.test_path)
    list_ds = tf.data.Dataset.list_files(test_list)
    num_threads = 5
    test_ds = list_ds.map(lambda x: load_image(x, resize, channels), num_parallel_calls=num_threads)
    return test_ds

def get_image_list(mode=str, data_path=str):
    images_path = data_path + "\\*"
    images_list = glob(images_path)
    final_images_list = []
    for image in images_list:
        if os.path.isfile(image) and any(image.lower().endswith(ext) for ext in const.IMAGE_EXTENSIONS):
            final_images_list.append(image)
        else:
            raise IOError(errno.EINVAL, os.strerror(errno.EINVAL), image)
    if final_images_list:
        return final_images_list
    else:
        raise CustomError(f"Input mode'{mode}' and given path : '{images_path}', "
                          f"are not compatible. Recheck input mode and input path Directory")

def load_image(input_data, resize, channels):
    img = tf.io.read_file(input_data)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size (total pixel count).
    input_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    size = tf.cast(resize, tf.float32)
    ratio = tf.math.divide(input_shape[1], input_shape[0])
    new_height = tf.math.sqrt(tf.math.divide(size, ratio))
    new_width = tf.math.divide(size, new_height)
    new_shape = tf.cast([new_height, new_width], tf.int32)
    img = tf.image.resize(img, new_shape)
    return img
