import tensorflow as tf


def down_sample_v1(input_layer, n_filters, name, batch_norm=False, pool=True,
                   activation=tf.keras.activations.relu, training=True):
    x = net = input_layer
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.keras.layers.Conv3D(
                F, (3, 3, 3),
                activation=None,
                padding='same',
                name="conv_{}".format(i + 1))(net)
            if batch_norm:
                net = tf.keras.layers.BatchNormalization(name="bn_{}".format(i + 1))(net, training=training)
            net = tf.keras.layers.Activation(activation, name="ac_{}".format(i + 1))(net)
    if not pool:
        return net

    pool = tf.keras.layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), name="pool_{}".format(name))(net)
    return net, pool


def down_sample(input_layer, n_filters, name,
                batch_norm=False, pool=True, activation=tf.keras.activations.relu,
                training=True):
    net = input_layer
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.keras.layers.Conv3D(
                F, (3, 3, 3),
                activation=None,
                padding='same',
                name="conv_{}".format(i + 1))(net)
            if batch_norm:
                net = tf.keras.layers.BatchNormalization(name="bn_{}".format(i + 1))(net, training=training)
            net = tf.keras.layers.Activation(activation, name="ac_{}".format(i + 1))(net)
    if not pool:
        return net

    pool = tf.keras.layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), name="pool_{}".format(name))(net)
    return net, pool


def up_sample(input_a, input_b, name, deconv=False):
    """
    up sample input_a, and concatenate the result with input b.
    now only up sample, others need to do, but max_pool_with_argmax only support 4D tensor.
    only support patch shape 2^n, other may raise error, need to crop input_b.
    :param input_a: lower resolution, need to take up-sample/de-convolution/un-pooling op.
    :param input_b: higher resolution, concatenate with up-sample result.
    :param name: name
    :return: tensor, shape equal to input_b.shape
    """
    shape = input_a.shape
    with tf.variable_scope("layer{}".format(name)):
        if deconv:
            net = tf.keras.layers.Conv3DTranspose(shape[-1].value, (3, 3, 3), strides=(2, 2, 2), padding='same',
                                                  name='up_sample_deconv')(input_a)
        else:
            net = tf.keras.layers.UpSampling3D(name='up_sample')(input_a)
        # print(net)
        # print('******************************')
        net = tf.keras.layers.Concatenate(name='concatenate')([input_b, net])
    return net


def simple_unet_3d(input_shape, n_cls=2, batch_norm=False, deconv=False, training=True):
    x = tf.keras.Input(shape=input_shape, name='input_patch')
    y = tf.keras.Input(shape=input_shape[:-1], name='input_gt', dtype=tf.uint8)
    with tf.variable_scope('down_sample'):
        l1, pool1 = down_sample(x, [32, 64], 'l1', batch_norm=batch_norm, training=training)
        l2, pool2 = down_sample(pool1, [64, 128], 'l2', batch_norm=batch_norm, training=training)
        l3, pool3 = down_sample(pool2, [128, 256], 'l3', batch_norm=batch_norm, training=training)
        l4 = down_sample(pool3, [256, 512], 'l4', pool=False)
    with tf.variable_scope('up_sample'):
        l5 = up_sample(l4, l3, 'l5_up_sample_concatenate', deconv)
        l5 = down_sample(l5, [256, 256], 'l5_convolution', pool=False)
        l6 = up_sample(l5, l2, 'l6_up_sample_concatenate', deconv)
        l6 = down_sample(l6, [128, 128], 'l6_convolution', pool=False)
        l7 = up_sample(l6, l1, 'l7_up_sample_concatenate', deconv)
        l7 = down_sample(l7, [64, 64], 'l7_convolution', pool=False)
    # wait to check result, for one patch, prob.sum(axis = -1) == 1
    with tf.variable_scope('final'):
        out = tf.keras.layers.Conv3D(n_cls, (1, 1, 1), name='cnn_as_fc')(l7)
        out = tf.keras.layers.Softmax()(out)

    # model = tf.keras.Model(inputs=x, outputs=out) # wrong with name, not support scope name.
    return {'pred': out, 'input_x': x, 'input_y': y}


def dense_u3d(input_shape, n_cls=2, training=True, batch_norm=True, deconv=False, refine=False):
    x = tf.keras.Input(shape=input_shape, name='input_patch')
    y = tf.keras.Input(shape=input_shape[:-1], name='input_gt', dtype=tf.uint8)
    # dense type: 0: no dense 1: linear dense 2: fully dense
    with tf.variable_scope('down_sample'):
        l1, pool1 = down_sample(x, [32, 64], 'l1', batch_norm=batch_norm, training=training)
        l2, pool2 = down_sample(pool1, [64, 128], 'l2', batch_norm=batch_norm, training=training)
        l3, pool3 = down_sample(pool2, [128, 256], 'l3', batch_norm=batch_norm, training=training)
        l4 = down_sample(pool3, [256, 512], 'l4', pool=False)
    with tf.variable_scope('up_sample'):
        l5 = up_sample(l4, l3, 'l5_up_sample_concatenate', deconv)
        l5 = down_sample(l5, [256, 256], 'l5_convolution', pool=False)
        l6 = up_sample(l5, l2, 'l6_up_sample_concatenate', deconv)
        l6 = down_sample(l6, [128, 128], 'l6_convolution', pool=False)
        l7 = up_sample(l6, l1, 'l7_up_sample_concatenate', deconv)
        l7 = down_sample(l7, [64, 64], 'l7_convolution', pool=False)
    # wait to check result, for one patch, prob.sum(axis = -1) == 1
    with tf.variable_scope('final'):
        out = tf.keras.layers.Conv3D(n_cls, (1, 1, 1), name='cnn_as_fc')(l7)
        out = tf.keras.layers.Softmax()(out)

    # model = tf.keras.Model(inputs=x, outputs=out) # wrong with name, not support scope name.
    return {'pred': out, 'input_x': x, 'input_y': y}

# if __name__ == '__main__':
#     batch_size = [20]
#     patch_shape = [32, 32, 32]
#     channels = [4]
#     input_shape = batch_size + patch_shape + channels
#     x = tf.placeholder(tf.float32, shape=input_shape)
#     model = simple_unet_3d(x, input_shape)
#     # with tf.Session() as sess:
#     #     init = tf.global_variables_initializer()
#     #     sess.run(init)

