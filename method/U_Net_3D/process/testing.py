import numpy as np
import tensorflow as tf
from ..model.unet import simple_unet_3d
from .pre_process import load_patch_for_test_one_subj, read_img, get_patches_according_to_indices
from .utils.patches import reconstruct_from_patches
import os
import time


def prepare_data_per_case(case, config, valid=True):
    # load patch
    file_in = config['data_file_valid'] if valid else config['data_file_valid']
    patches, indices = load_patch_for_test_one_subj(file_in,
                                                    case,
                                                    config['patch_shape'])
    # print(patches.shape, indices.shape, time.time()-t)
    # shuffle and transpose [N, C, dim0, dim1, dim2] to [N, dim0, dim1, dim2, C]
    patches = np.transpose(patches, [0, 2, 3, 4, 1])
    return patches, indices


def test_on_full_images(sess, x, pred, config, debug=False):
    test_dice = []
    for sub_i in config['valid_case_list']:
        gt_name = os.path.join(config["data_file_valid"], str(sub_i), 'OT.nii.gz')
        img_gt = read_img(gt_name)
        patches_pred = []
        patches, indices = prepare_data_per_case(sub_i, config)
        data_shape = (config["n_labels"],) + img_gt.shape[-3:]
        if debug:
            ...
            continue
        batch_size = config['validation_batch_size']
        num_steps = np.ceil(patches.shape[0] / batch_size)
        for step in range(int(num_steps)):
            batch_x = patches[step * batch_size: (step + 1) * batch_size]
            batch_pred = sess.run(pred, feed_dict={x: batch_x})
            patches_pred.append(batch_pred)
        patches_pred = np.concatenate(patches_pred)
        patches_pred = np.transpose(patches_pred, [0, 4, 1, 2, 3])
        # print('after transpose:', patches_pred[:, 1].mean(), patches_pred[:, 1].max())
        # print('patch after argmax sum-cls1:', patches_pred.argmax(axis=1).sum())
        prob_map = reconstruct_from_patches(patches_pred, indices, data_shape)
        img_pred = np.argmax(prob_map, axis=0)
        dsc = 2 * np.sum(img_gt * img_pred) / (img_gt.sum() + img_pred.sum())
        test_dice.append(dsc)
        print('dice, gt.sum(), pred.sum():', dsc, img_gt.sum(), img_pred.sum())
    print('test mean dice:', np.mean(test_dice))


def test_process(config, debug=False):
    if debug:
        test_debug(config)
        return
    tf.reset_default_graph()
    input_shape = config["patch_shape"] + (config["n_channels"],)
    model = simple_unet_3d(input_shape, n_cls=config["n_labels"], batch_norm=config['batch_norm'])
    pred = model['pred']
    x = model['input_x']
    train_logdir = config['ckp_file']
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(train_logdir))
        print('After load, uninitialized variable num:', len(sess.run(tf.report_uninitialized_variables())))
        test_on_full_images(sess, x, pred, config)


def test_debug(config):
    test_on_full_images(None, None, None, config, debug=True)
    return



