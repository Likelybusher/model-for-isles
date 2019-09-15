import numpy as np
import tensorflow as tf
from ..model.unet import simple_unet_3d
from ..metrics import cross_entropy_loss_with_weight, dice_coef, cross_entropy_loss, cross_entropy_loss_v1
from .pre_process import load_patch_for_epoch
from .utils .augmentation import augment_patch
from .testing import test_on_full_images
import os
import time


def calculate_weight(y_true, n_cls):
    num_pixels = np.product(y_true.shape)
    num_per_class = np.bincount(y_true.flatten(), minlength=n_cls)
    return 1.0 / num_per_class * num_pixels / n_cls


def prepare_data(config, valid=False):
    # load patch
    if valid:
        x_train_epoch, y_train_epoch = load_patch_for_epoch(config['data_file_valid'],
                                                            config['valid_case_list'],
                                                            config['patch_shape'])
    else:
        x_train_epoch, y_train_epoch = load_patch_for_epoch(config['data_file_train'],
                                                            config['train_case_list'],
                                                            config['patch_shape'])
        # argument
        for i in range(x_train_epoch.shape[0]):
            x_train_epoch[i], y_train_epoch[i] = augment_patch(x_train_epoch[i], y_train_epoch[i], config['augment'])

    # shuffle and transpose [N, C, dim0, dim1, dim2] to [N, dim0, dim1, dim2, C]
    index = np.arange(x_train_epoch.shape[0])
    np.random.shuffle(index)
    x_train_epoch, y_train_epoch = [x_train_epoch[index], y_train_epoch[index]]
    x_train_epoch = np.transpose(x_train_epoch, [0, 2, 3, 4, 1])
    # calculate class weight for loss.
    weight = calculate_weight(y_train_epoch.astype(np.int32), config['n_labels'])
    return x_train_epoch, y_train_epoch, weight


def test_function(config):
    x_train_epoch, y_train_epoch, weight = prepare_data(config, valid=False)
    print(weight)
    # for i in range(y_train_epoch.shape[0]):
    #     print(y_train_epoch[i].sum(), 40*40*40-y_train_epoch[i].sum())


def train_process(config, debug=False, ck_name='ckpt'):
    if debug:
        test_function(config)
        return

    tf.reset_default_graph()
    # create or load model
    input_shape = config["patch_shape"] + (config["n_channels"],)
    model = simple_unet_3d(input_shape, n_cls=config["n_labels"], batch_norm=config['batch_norm'],
                           deconv=config['deconv'])
    pred = model['pred']
    x, y = [model['input_x'], model['input_y']]
    #
    with tf.variable_scope('loss'):
        loss = cross_entropy_loss(y, pred)
        # weight = tf.placeholder(dtype=tf.float32, shape=[config['n_labels']], name='weight_for_loss')
        # loss = cross_entropy_loss_with_weight(y, pred)  # weight calculated per batch.
        # loss = cross_entropy_loss_with_weight(y, pred, weight)  # weight calculated per epoch.
        # dice = dice_coef(y, pred)
    ops = tf.get_default_graph().get_operations()
    update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
    for op in update_ops:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)
        # print('add to collection:', op.name, op.type)
    with tf.control_dependencies(update_ops):
        training_op = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
    print('Model established.')
    # no circle but raise error, may caused by CUDA.
    # https://github.com/tensorflow/tensorflow/issues/24816
    # g = tf.get_default_graph()
    # print(g)
    # writer = tf.summary.FileWriter(os.path.abspath('./log_board'), g)
    # print(os.path.abspath('./log_board'))
    train_logdir = config['ckp_file']
    if not os.path.exists(train_logdir):
        os.mkdir(train_logdir)

    # init Saver
    # save nothing when use tf.train.Checkpoint, so use tf.train.Saver instead.
    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += tf.trainable_variables()
    print('var saved list:', var_list)
    saver = tf.train.Saver(var_list=var_list)
    save_path_prefix = os.path.join(train_logdir, ck_name)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Init completed.')
        for epoch in range(config["n_epochs"]):
            x_train_epoch, y_train_epoch, weight_train_epoch = prepare_data(config)
            # print(weight_train_epoch)
            num_steps = len(y_train_epoch) // config['batch_size']
            # print('start new epoch %d!' % epoch)
            train_loss_epoch = []
            train_y_pred_sum_epoch = []
            for step in range(num_steps):
                batch_x = x_train_epoch[step * config['batch_size']: (step + 1) * config['batch_size']]
                batch_y = y_train_epoch[step * config['batch_size']: (step + 1) * config['batch_size']]
                batch_weight = weight_train_epoch
                assert len(batch_x) == config['batch_size']
                _, train_loss_step, batch_y_pred = sess.run([training_op, loss, pred],
                                                            feed_dict={x: batch_x, y: batch_y})
                # _, train_loss_step, batch_y_pred = sess.run([training_op, loss, pred],
                #                                             feed_dict={x: batch_x, y: batch_y, weight: batch_weight})
                train_loss_epoch.append(train_loss_step)
                train_y_pred_sum_epoch.append(batch_y_pred.argmax(axis=-1).sum())
            if not epoch % 10:
                print('train loss on epoch %d:' % epoch, np.mean(train_loss_epoch))
            # if not epoch % config["valid_every_n_epochs"] and epoch:
            #     test_on_full_images(sess, x, pred, config)
            if not epoch % 50 and epoch:
                saver.save(sess, save_path_prefix, global_step=epoch)
                # test_on_full_images(sess, x, pred, config, debug=True)

        print('training completed.')
        saver.save(sess, save_path_prefix, global_step=config["n_epochs"])
