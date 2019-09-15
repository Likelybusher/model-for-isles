import tensorflow as tf
import numpy as np


def cross_entropy_loss_v1(y_true, y_pred, sample_weight=None, eps=1e-6):
    """
    :param y_pred: output 5D tensor, [batch size, dim0, dim1, dim2, class]
    :param y_true: 4D GT tensor, [batch size, dim0, dim1, dim2]
    :param eps: avoid log0
    :return: cross entropy loss
    """
    log_y = tf.log(y_pred + eps)
    num_samples = tf.cast(tf.reduce_prod(tf.shape(y_true)), "float32")
    label_one_hot = tf.one_hot(indices=y_true, depth=y_pred.shape[-1], axis=-1, dtype=tf.float32)
    if sample_weight is not None:
        # ce = mean(- weight * y_true * log(y_pred)).
        label_one_hot = label_one_hot * sample_weight
    cross_entropy = - tf.reduce_sum(label_one_hot * log_y) / num_samples
    return cross_entropy


def cross_entropy_loss(y_true, y_pred, sample_weight=None):
    # may not use one_hot when use tf.keras.losses.CategoricalCrossentropy
    y_true = tf.one_hot(indices=y_true, depth=y_pred.shape[-1], axis=-1, dtype=tf.float32)
    if sample_weight is not None:
        # ce = mean(weight * y_true * log(y_pred)).
        y_true = y_true * sample_weight
    # for tf version 1.9.0
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    # for tf version 1.13.0
    # return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def cross_entropy_loss_with_weight(y_true, y_pred, sample_weight_per_c=None, eps=1e-6):
    # for simple calculate this batch.
    # if possible, get weight per epoch before training.
    num_dims, num_classes = [len(y_true.shape), y_pred.shape.as_list()[-1]]
    if sample_weight_per_c is None:
        print('use batch to calculate weight')
        num_lbls_in_ygt = tf.cast(tf.reduce_prod(tf.shape(y_true)), dtype="float32")
        num_lbls_in_ygt_per_c = tf.bincount(arr=tf.cast(y_true, tf.int32), minlength=num_classes, maxlength=num_classes,
                                            dtype="float32")  # without the min/max, length of vector can change.
        sample_weight_per_c = (1. / (num_lbls_in_ygt_per_c + eps)) * (num_lbls_in_ygt / num_classes)
    sample_weight_per_c = tf.reshape(sample_weight_per_c, [1] * num_dims + [num_classes])
    # use cross_entropy_loss get negative value, while cross_entropy_loss and cross_entropy_loss_v1 get the same
    # when no weight. I guess may some error when batch distribution is huge different from epoch distribution.
    return cross_entropy_loss_v1(y_true, y_pred, sample_weight=sample_weight_per_c)


def dice_coef(y_true, y_pred, eps=1e-6):
    # problem: when gt class-0 >> class-1, the pred p(class-0) >> p(class-1)
    # eg. gt = [0, 0, 0, 0, 1] pred = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]. 2 * 4 / (5 + 5) = 0.8
    # in fact, change every pred, 4/5 -> 0.6, 1/5 ->1, so the model just pred all 0. imbalance class problem.
    # only calculate gt == 1 can fix my problem, but for multi-class task, weight needed like ce loss above.
    y_true = tf.one_hot(indices=y_true, depth=y_pred.shape[-1], axis=-1, dtype=tf.float32)
    abs_x_and_y = 2 * tf.reduce_sum(y_true * y_pred)
    abs_x_plus_abs_y = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (abs_x_and_y + eps) / (abs_x_plus_abs_y + eps)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def test_metric(func, y_true, y_pred, feed_dict, weight=None):
    with tf.Session() as sess:
        # print(sess.run(..., feed_dict=feed_dict))
        print(sess.run(func(y_true, y_pred, weight), feed_dict=feed_dict))


def test():
    tf.reset_default_graph()
    y_true = tf.placeholder(dtype=tf.uint8, shape=(None, ))
    y_pred = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    weight = tf.placeholder(dtype=tf.float32, shape=(2, ))
    y_true_batch = np.array([1, 1, 1, 1, 0])
    y_pred_batch = np.array([
                             [0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])

    feed_dict = {y_true: y_true_batch, y_pred: y_pred_batch}
    test_metric(cross_entropy_loss, y_true, y_pred, feed_dict)
    test_metric(cross_entropy_loss_v1, y_true, y_pred, feed_dict)
    # test_metric(cross_entropy_loss_with_weight, y_true, y_pred, feed_dict)
    # feed_dict[weight] = [5, 1/2]
    # test_metric(cross_entropy_loss_with_weight, y_true, y_pred, feed_dict, weight)


if __name__ == '__main__':
    test()
