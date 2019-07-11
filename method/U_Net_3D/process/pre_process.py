from __future__ import absolute_import, print_function, division
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from .utils.patches import compute_patch_indices


def load_patch_for_test_one_subj(file_path, sub_i, patch_shape, over_lap=10,
                                 modalities=['MR_DWI', 'MR_Flair', 'MR_T1', 'MR_T2'],
                                 mask_sym='MR_MASK',
                                 suffix='.nii.gz',
                                 use_norm=True):
    """
    for test, split the full image, similar to load_patch_for_epoch, may merge to one function.
    :param file_path:
    :param sub_i:
    :param patch_shape:
    :param over_lap:
    :param modalities:
    :param mask_sym:
    :param suffix:
    :param use_norm:
    :return:
    """

    # first load image, mask.
    mask_name = os.path.join(file_path, str(sub_i), mask_sym + suffix)
    if os.path.exists(mask_name):
        mask_img = read_img(mask_name)
    channels_one_sub = []
    for i in range(len(modalities)):
        img_name = os.path.join(file_path, str(sub_i), modalities[i] + suffix)
        img_name_norm = os.path.join(file_path, str(sub_i), modalities[i] + '_norm' + suffix)
        if not os.path.exists(img_name):
            raise Exception('cannot find the path %s!' % img_name)
        if not os.path.exists(img_name_norm):  # may raise error but my data has mask.
            write_norm_img(img_name, img_name_norm, mask_img)
        elif use_norm:  # if exist norm data and use, replace to load it.
            img_name = img_name_norm
        channels_one_sub.append(read_img(img_name))

    if not os.path.exists(mask_name):
        mask_img = np.ones(shape=channels_one_sub[0].shape, dtype=np.float)
    channels_one_sub.append(mask_img)
    channels_one_sub = np.asarray(channels_one_sub)
    # second sample patch.
    indices = compute_patch_indices(channels_one_sub[0].shape, patch_shape, over_lap)

    patches, chosen = get_patches_according_to_indices(channels_one_sub, patch_shape, np.transpose(indices), True, True)
    indices = indices[chosen]

    return np.asarray(patches[:, :len(modalities)]), np.asarray(indices)


def load_patch_for_epoch(file_path, subject_list, patch_shape, num_sample=50,
                         modalities=['MR_DWI', 'MR_Flair', 'MR_T1', 'MR_T2'],
                         mask_sym='MR_MASK',
                         gt_sym='OT',
                         suffix='.nii.gz',
                         use_norm=True):
    channels = []  # (n_sub, n_channels, n, )
    gt = []
    for sub_i in subject_list:
        # first load image, mask, and gt.
        mask_name = os.path.join(file_path, str(sub_i), mask_sym + suffix)
        if os.path.exists(mask_name):
            mask_img = read_img(mask_name)
        channels_one_sub = []
        for i in range(len(modalities)):
            img_name = os.path.join(file_path, str(sub_i), modalities[i] + suffix)
            img_name_norm = os.path.join(file_path, str(sub_i), modalities[i] + '_norm' + suffix)
            if not os.path.exists(img_name):
                raise Exception('cannot find the path %s!' % img_name)
            if not os.path.exists(img_name_norm):  # may raise error but my data has mask.
                write_norm_img(img_name, img_name_norm, mask_img)
            elif use_norm:  # if exist norm data and use, replace to load it.
                img_name = img_name_norm
            channels_one_sub.append(read_img(img_name))

        if not os.path.exists(mask_name):
            mask_img = np.ones(shape=channels_one_sub[0].shape, dtype=np.float)
        channels_one_sub.append(mask_img)

        gt_name = os.path.join(file_path, str(sub_i), gt_sym + suffix)
        channels_one_sub.append(read_img(gt_name))
        # second sample patch.
        indices = get_random_patch_indices(channels_one_sub[0].shape, patch_shape,
                                           channels_one_sub[-2], channels_one_sub[-1])
        patches, _ = get_patches_according_to_indices(channels_one_sub, patch_shape, indices)
        for patch in patches:
            channels.append(patch[:len(modalities)])
            gt.append(patch[-1])
    return np.asarray(channels), np.asarray(gt)


def read_img(in_file):
    itk_img = sitk.ReadImage(in_file)
    img_array = sitk.GetArrayFromImage(itk_img)

    return img_array


def write_norm_img(in_file, out_file, mask_img):
    # norm the masked img.
    itk_img = sitk.ReadImage(in_file)
    img_array = sitk.GetArrayFromImage(itk_img)
    img_array_masked = img_array * mask_img
    cnt = mask_img.sum()
    img_mean = img_array_masked.sum() / cnt
    img_std = np.sqrt(((img_array_masked - img_mean) * (img_array_masked - img_mean) * mask_img).sum()/ cnt)
    img_array_masked = (img_array_masked - img_mean) / img_std
    itk_img_masked = sitk.GetImageFromArray(img_array_masked)
    itk_img_masked.CopyInformation(itk_img)
    sitk.WriteImage(itk_img_masked, out_file)


def show(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def get_random_patch_indices(image_shape, patch_shape, mask, gt, n_class=2, num_sample=50):
    sample_shape = np.subtract(image_shape, patch_shape) + 1
    cls_indices = []
    # the sample point is the corner, weight map fixed to center by shift.
    shift = np.asarray(patch_shape) // 2
    for i in range(n_class):
        mask_ = mask[shift[0]:shift[0]+sample_shape[0],
                     shift[1]:shift[1]+sample_shape[1],
                     shift[2]:shift[2]+sample_shape[2]]
        gt_ = gt[shift[0]:shift[0]+sample_shape[0],
                 shift[1]:shift[1]+sample_shape[1],
                 shift[2]:shift[2]+sample_shape[2]] == i
        weight_map = gt_ * mask_
        weight_map /= weight_map.sum()
        cls_indices.append(get_random_nd_indices(sample_shape, weight_map, num_sample//n_class))
    return np.concatenate(cls_indices, axis=1)


def get_random_nd_indices(sample_shape, weight_map, num_sample=50):
    indices_flatten = np.random.choice(np.product(sample_shape),
                                       size=num_sample,
                                       replace=True,
                                       p=weight_map.flatten())
    indices = np.asarray(np.unravel_index(indices_flatten, weight_map.shape))
    return indices


def get_patches_according_to_indices(array, patch_shape, indices, padding=False, roi=False):
    """
    faster than method in utils.patches, only one padding op.
    :param array: image array shape: (n_channels, dim0, dim1, dim2)
    :param patch_shape:
    :param indices:
    :param padding: if true, padding the array.
    :param roi: if true, only patch in roi will be chosen.
    :return:
    """
    samples = []
    samples_indices = []
    array = np.asarray(array)
    if padding:
        pad_width = np.asarray((0, ) + patch_shape) // 2
        pad_width = np.stack([pad_width, pad_width], axis=1)
        array = np.pad(array, pad_width, mode='edge')
        indices_shift = np.asarray(patch_shape)[:, np.newaxis] // 2
        indices = indices + np.tile(indices_shift, indices.shape[1])

    for i in range(indices.shape[1]):
        patch = array[:,
                      indices[0, i]: indices[0, i]+patch_shape[0],
                      indices[1, i]: indices[1, i]+patch_shape[1],
                      indices[2, i]: indices[2, i]+patch_shape[2]]
        if roi and not patch[-1].sum():   # in test process, mask is the last channel
            continue

        samples.append(patch)
        samples_indices.append(i)
    return np.asarray(samples), samples_indices  # (n_samples, channels, dim0, dim1, dim2)
