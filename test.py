import os
import cv2
import numpy as np
import tensorflow as tf
import constants as const


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


def load_image(input_data: str, size: int, channels: int):
    basename = os.path.basename(input_data).split('.')[0]
    original_img = cv2.imread(input_data)
    if channels == 1:
        test_img = cv2.imread(input_data, 0)
    else:
        test_img = original_img
    input_shape = original_img.shape[:2]
    ratio = input_shape[1]/input_shape[0]
    new_height = int(np.sqrt(size / ratio))
    new_width = int(size / new_height)
    print("original shape", input_shape)

    test_img = cv2.resize(test_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print("new shape", test_img.shape)
    return test_img, original_img, basename


def test(config, image_ds, model_dir):
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.visible_device_list = config.session.gpu
    with tf.Session(graph=tf.Graph(), config=session_conf) as sess:
        loaded_model = tf.saved_model.loader.load(sess, [const.SERVING_SESSION], model_dir)
        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def[const.SERVING_SIGNATURE])
        _input_tensor = input_dict[const.INPUT_TENSOR_KEY]
        _output_dict = output_dict[const.OUTPUT_TENSOR_KEY]

        channels = _input_tensor.shape[-1]
        resize = config.data_prep.resize

        for image in image_ds:
            prediction_outputs = sess.run(_output_dict, feed_dict={_input_tensor: image})

            '''
            prediction_outputs[0, :, :, :] are the H*W*classes for each image

            The model that is used for this is tensorflow serving model, a .pb file and a folder for variables
            const.SERVING_SESSION
            const.SERVING_SIGNATURE
            const.INPUT_TENSOR_KEY
            const.OUTPUT_TENSOR_KEY
            These four parameters are to be manually defined when exporting the tf model from checkpoints

            Further post processing steps like binarization and or class assignment etc should come after this

            '''
