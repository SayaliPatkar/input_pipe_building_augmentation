import os
import yamlargparse
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import image_loader as loader
import test

class CustomError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def get_parser():
    parser = yamlargparse.ArgumentParser(
        prog='custom_post_processing',
        description='Post processing configuration for visualization, baselines and height segmentation'
    )

    parser.add_argument('--input.train_path', default='', help='relative path to the folder containing training data')
    parser.add_argument('--input.val_path', default='', help='relative path to the folder containing validation data')
    parser.add_argument('--input.test_path', default='', help='relative path to the folder containing test data')
    parser.add_argument('--input.model_path', help='the relative path to the model for prediction')

    parser.add_argument('--data_prep.mode', default=const.AUG_DEMO, choices=[const.TEST, const.TRAIN, const.AUG_DEMO],
                    help='can take two values train or test')
    parser.add_argument('--data_prep.channels', default=const.COLOR_IMAGE_CHANNLES,
                        choices=[const.COLOR_IMAGE_CHANNLES, const.GREY_IMAGE_CHANNELS],
                        help='color or greyscale image')
    parser.add_argument('--data_prep.resize', default=0, help='the resized input dimensions H*W')
    parser.add_argument('--data_prep.augmentation', action=yamlargparse.ActionYesNo, default=False,
                        help='the choice to perform augmentation')

    parser.add_argument('--augmentation_ops.color',  action=yamlargparse.ActionYesNo, default=False,
                        help='the choice to perform color augmentation')
    parser.add_argument('--augmentation_ops.gaussian_noise',  action=yamlargparse.ActionYesNo, default=False,
                        help='the choice to add Gaussian noise to the image')
    parser.add_argument('--augmentation_ops.flip_vert',  action=yamlargparse.ActionYesNo, default=False,
                        help='the choice to perform vertical flips')
    parser.add_argument('--augmentation_ops.flip_horz',  action=yamlargparse.ActionYesNo, default=False,
                        help='the choice to perform horizontal flips')
    parser.add_argument('--augmentation_ops.max_rotation', default=0,
                        help='maximum rotation permissible during augmentation')
    parser.add_argument('--augmentation_ops.max_scaling', default=0,
                        help='maximum scaling permissible during augmentation')

    parser.add_argument('--session.gpu', default='0')
    parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile)
    return parser


def plot_images(augmented_arr):
    cols = ['Original','Vert flip','Horz flip','Scaled', 'Color', 'Noise', 'Rotation']
    fig, axes = plt.subplots(nrows=8, ncols=7, figsize=(20, 20))
    fig.patch.set_visible(False)
    counter = 1
    for n, col in enumerate(cols):
        ax = plt.subplot(8,7,counter)
        ax.axis('off')
        ax.text(0.3,0.2, col, fontsize=12)
        counter += 1

    for cluster in augmented_arr:
        for n in range(7):
         ax = plt.subplot(8,7,counter)
         ax.axis('off')
         plt.imshow(cluster[n].numpy(), interpolation='nearest')
         counter += 1
    fig.tight_layout()
    plt.show()


def run():
    parser = get_parser()
    config = parser.parse_args(['--cfg', './main_config.yaml'])
    if config.data_prep.mode == 'augmentation_demo':
        # get augmented tf dataset
        augmented_arr = loader.prepare_input_pipeline(config)
        plot_images(augmented_arr)
    if config.data_prep.mode == 'train':
        # get augmented tf dataset
        train_ds = loader.prepare_input_pipeline(config)
        #further batching and training
    if config.data_prep.mode == 'test':
        # load test dataset, no augmentation
        test_ds = loader.prepare_input_pipeline(config)
        path_net_pb = config.input.model_path
        model_dir = get_model_file(path_net_pb)
        test_code.test(config, image_dict=im_dict, model_dir=model_dir)

if __name__ == '__main__':
    run()
