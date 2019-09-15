import numpy as np
import tensorflow as tf
from ..model.unet import simple_unet_3d
from .pre_process import load_patch_for_test_one_subj
from .utils.patches import reconstruct_from_patches
import os
import SimpleITK as sitk
import time


def prepare_data_per_case(case, config, valid=True, infer=False):
    # load patch
    file_in = config['data_file_valid'] if valid else config['data_file_valid']
    if infer:
        file_in = config["data_file_infer"]
    patches, indices = load_patch_for_test_one_subj(file_in,
                                                    case,
                                                    config['patch_shape'])
    # print(patches.shape, indices.shape, time.time()-t)
    # shuffle and transpose [N, C, dim0, dim1, dim2] to [N, dim0, dim1, dim2, C]
    patches = np.transpose(patches, [0, 2, 3, 4, 1])
    return patches, indices


def test_on_full_images(sess, x, pred, config, debug=False, valid=True, save=False):
    test_dice = []
    case_list = config['valid_case_list'] if valid else config['test_case_list']
    if debug:
        case_list = config['train_case_list']
    print('test case list', case_list)
    for sub_i in case_list:
        ref_name = os.path.join(config["data_file_valid"], str(sub_i), 'OT.nii.gz')
        pre_name = os.path.join(config["data_file_valid"], str(sub_i), 'pred.nii.gz')
        ref_img = sitk.ReadImage(ref_name)
        ref_img_array = sitk.GetArrayFromImage(ref_img)
        patches_pred = []
        patches, indices = prepare_data_per_case(sub_i, config)
        data_shape = (config["n_labels"],) + ref_img_array.shape[-3:]
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
        img_pred_array = np.argmax(prob_map, axis=0)
        dsc = 2 * np.sum(ref_img_array * img_pred_array) / (ref_img_array.sum() + img_pred_array.sum())
        test_dice.append(dsc)
        print('dice, gt.sum(), pred.sum():', dsc, ref_img_array.sum(), img_pred_array.sum())
        if save:
            img_pred_array.astype('uint8')
            pred_img = sitk.GetImageFromArray(img_pred_array)
            pred_img.CopyInformation(ref_img)
            sitk.WriteImage(pred_img, pre_name)

    print('test mean dice:', np.mean(test_dice))


def infer_on_full_images(sess, x, pred, config):
    file_in = config["data_file_infer"]
    file_out = os.path.join(config["data_file_infer"][:-4], 'pred')
    if not os.path.exists(file_out):
        os.makedirs(file_out)

    case_indices = os.listdir(file_in)
    for case_i in case_indices:
        ref_name = os.path.join(file_in, case_i, 'MR_Flair.nii.gz')
        ref_img = sitk.ReadImage(ref_name)
        ref_img_array = sitk.GetArrayFromImage(ref_img)
        patches_pred = []
        patches, indices = prepare_data_per_case(case_i, config, infer=True)
        data_shape = (config["n_labels"],) + ref_img_array.shape[-3:]
        batch_size = config['validation_batch_size']
        num_steps = np.ceil(patches.shape[0] / batch_size)
        for step in range(int(num_steps)):
            batch_x = patches[step * batch_size: (step + 1) * batch_size]
            batch_pred = sess.run(pred, feed_dict={x: batch_x})
            patches_pred.append(batch_pred)
        patches_pred = np.concatenate(patches_pred)
        patches_pred = np.transpose(patches_pred, [0, 4, 1, 2, 3])
        prob_map = reconstruct_from_patches(patches_pred, indices, data_shape)
        img_pred_array = np.argmax(prob_map, axis=0)
        img_pred_array.astype('uint8')
        pre_file = os.path.join(file_out, case_i)
        if not os.path.exists(pre_file):
            os.makedirs(pre_file)
        pre_name = os.path.join(pre_file, 'pred.nii.gz')
        pred_img = sitk.GetImageFromArray(img_pred_array)
        pred_img.CopyInformation(ref_img)
        sitk.WriteImage(pred_img, pre_name)

        print(case_i, 'pred.sum():', img_pred_array.sum())


def test_process(config, infer=False, global_step=0, ck_name='ckpt'):
    tf.reset_default_graph()
    input_shape = config["patch_shape"] + (config["n_channels"],)
    model = simple_unet_3d(input_shape, n_cls=config["n_labels"], batch_norm=config['batch_norm'], training=False,
                           deconv=config['deconv'])
    pred = model['pred']
    x = model['input_x']
    train_logdir = config['ckp_file']

    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += tf.trainable_variables()
    saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
    if not global_step:
        ckp_dir = tf.train.latest_checkpoint(train_logdir)
    else:
        ckp_dir = train_logdir + '/' + ck_name + '-' + str(global_step)
    print('restore ckp dir:', ckp_dir)
    with tf.Session() as sess:
        saver.restore(sess, ckp_dir)
        print('After load, uninitialized variable num:', len(sess.run(tf.report_uninitialized_variables())))
        if not infer:
            test_on_full_images(sess, x, pred, config, debug=False, valid=False)
            test_on_full_images(sess, x, pred, config, debug=True, valid=False)
        else:
            infer_on_full_images(sess, x, pred, config)


def test_debug(config):

    return



