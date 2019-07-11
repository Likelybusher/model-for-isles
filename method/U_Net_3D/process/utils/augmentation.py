# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import numpy as np


def augment_patch(channels, gt_lbls, params=None):
    # modified, only one pathway.
    # channels: np arrays [channels, x, y, z].
    # gt_lbls: np array of shape [x,y,z]
    # params: None or Dictionary, with params of each augmentation type. }
    # data argument for 3d images.
    if params is not None:
        channels = random_histogram_distortion(channels, params['hist_dist'])
        channels, gt_lbls = random_flip(channels, gt_lbls, params['reflect'])
        channels, gt_lbls = random_rotation_90(channels, gt_lbls, params['rotate90'])
        
    return channels, gt_lbls


def random_histogram_distortion(channels, params):
    # Shift and scale the histogram of each channel.
    # channels: np arrays [channels, x, y, z].
    # params: { 'shift': {'mu': 0.0, 'std':0.}, 'scale':{'mu': 1.0, 'std': '0.'} }
    if params is None:
        return channels
    
    n_channs = channels.shape[0]
    if params['shift'] is None:
        shift_per_chan = 0.
    elif params['shift']['std'] != 0:  # np.random.normal does not work for an std==0.
        shift_per_chan = np.random.normal(params['shift']['mu'], params['shift']['std'], [n_channs, 1, 1, 1])
    else:
        shift_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * params['shift']['mu']
    
    if params['scale'] is None:
        scale_per_chan = 1.
    elif params['scale']['std'] != 0:
        scale_per_chan = np.random.normal(params['scale']['mu'], params['scale']['std'], [n_channs, 1, 1, 1])
    else:
        scale_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * params['scale']['mu']
    
    # Intensity augmentation
    channels = (channels + shift_per_chan) * scale_per_chan
        
    return channels


def random_flip(channels, gt_lbls, probs_flip_axes=None):
    # Flip (reflect) along each axis.
    # channels: np arrays [channels, x, y, z].
    # gt_lbls: np array of shape [x,y,z]
    # probs_flip_axes: list of probabilities, one per axis.
    if probs_flip_axes is None:
        return channels, gt_lbls
    
    for axis_idx in range(len(gt_lbls.shape)):
        flip = np.random.choice(a=(True, False), size=1, p=(probs_flip_axes[axis_idx], 1. - probs_flip_axes[axis_idx]))
        if flip:
            channels = np.flip(channels, axis=axis_idx+1)  # + 1 because dim [0] is channels.
            gt_lbls = np.flip(gt_lbls, axis=axis_idx)
            
    return channels, gt_lbls


def random_rotation_90(channels, gt_lbls, probs_rot_90=None):
    # Rotate by 0/90/180/270 degrees.
    # np arrays [channels, x, y, z].
    # gt_lbls: np array of shape [x,y,z]
    # probs_rot_90: {'xy': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'yz': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'xz': {'0': fl, '90': fl, '180': fl, '270': fl} }
    if probs_rot_90 is None:
        return channels, gt_lbls
        
    for key, plane_axes in zip(['xy', 'yz', 'xz'], [(0, 1), (1, 2), (0, 2)]):
        probs_plane = probs_rot_90[key]
        
        if probs_plane is None:
            continue
        
        assert len(probs_plane) == 4  # rotation 0, rotation 90 degrees, 180, 270.
        assert channels.shape[1+plane_axes[0]] == channels.shape[1+plane_axes[1]]  # Image/patch must be isotropic.
        
        # Normalize probs
        sum_p = probs_plane['0'] + probs_plane['90'] + probs_plane['180'] + probs_plane['270']
        if sum_p == 0:
            continue
        for rot_k in probs_plane:
            probs_plane[rot_k] /= sum_p  # normalize p to 1.
            
        p_rot_90_x0123 = (probs_plane['0'], probs_plane['90'], probs_plane['180'], probs_plane['270'])
        rot_90_xtimes = np.random.choice(a=(0, 1, 2, 3), size=1, p=p_rot_90_x0123)
        channels = np.rot90(channels, k=rot_90_xtimes, axes=[axis+1 for axis in plane_axes])
        gt_lbls = np.rot90(gt_lbls, k=rot_90_xtimes, axes=plane_axes)
        
    return channels, gt_lbls

