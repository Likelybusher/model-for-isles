import os
from U_Net_3D.process.training import train_process
from U_Net_3D.process.testing import test_process
import numpy as np


def get_config():
    config = dict()
    # INPUT CONFIG
    config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
    config["patch_shape"] = (40, 40, 40)  # switch to None to train on the whole image
    config["labels"] = (0, 1)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["all_modalities"] = ["MR_DWI", "MR_Flair", "MR_T1", "MR_T2"]
    config["training_modalities"] = config["all_modalities"]
    config["n_channels"] = len(config["training_modalities"])
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["n_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))
    # NET CONFIG
    config["deconvolution"] = False  # if False, will use up-sampling instead of deconvolution
    config['batch_norm'] = True
    # TRAIN CONFIG
    config['device'] = '/gpu:1'
    config["batch_size"] = 10
    config["validation_batch_size"] = 20
    config["n_epochs"] = 300  # cutoff the training after this many epochs
    config["valid_every_n_epochs"] = 20  # full image valid every n epochs
    config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    config["augment"] = {'hist_dist': {'shift': {'mu': 0., 'std': 0.05}, 'scale': {'mu': 1., 'std': 0.01}},
                         'reflect': (0.1, 0.1, 0.1),
                         'rotate90': {'xy': {'0': 0.8, '90': 0.05, '180': 0.1, '270': 0.05},
                                      'yz': {'0': 0.8, '90': 0.05, '180': 0.1, '270': 0.05},
                                      'xz': {'0': 0.8, '90': 0.05, '180': 0.1, '270': 0.05}
                                      }
                         }
    config["num_train_case"] = 28
    config["data_file_train"] = os.path.abspath("../data/IS2015/train")
    config["data_file_valid"] = os.path.abspath("../data/IS2015/train")
    config["train_case_list"] = list(range(1, config["num_train_case"] + 1))
    np.random.shuffle(config["train_case_list"])
    config["train_case_list"], config["valid_case_list"] = [config["train_case_list"][:-4],
                                                            config["train_case_list"][-4:]]
    # [21, 20, 14, 2, 7, 16, 27, 25, 9, 4, 5, 6, 17, 28, 11, 10, 15, 12, 24, 19, 18, 8, 13, 22]
    # [23, 1, 3, 26]
    # [5, 6, 15, 12] [19, 18, 12, 3] [15, 17, 19, 22]
    print('train case list:', config["train_case_list"])
    print('valid case list:', config["valid_case_list"])
    config["model_file"] = os.path.abspath("U_Net_3D")
    config["ckp_file"] = config["model_file"] + '/ckp'
    return config


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_process(config)
    # train_process(config, debug=True)
    test_process(config)
    pass


main(config=get_config())
