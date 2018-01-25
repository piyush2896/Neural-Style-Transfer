import tensorflow as tf
import pprint


def vgg19(is_input_trainable=False, fine_tune_last=False,
          n_classes=1000, input_shape=[None, 224, 224, 3],
          n_last_layers_trainable=0):
    path_conv = 'vgg_19/conv'
    path_fc = 'vgg_19/fc'
    ckpt_path = './pretrained-model/vgg19/vgg_19.ckpt'
    file = tf.train.NewCheckpointReader(ckpt_path)

    def _weights(stage, block=None, type_code=0):
        if type_code == 0:
            path = path_conv + str(stage) + '/conv' + str(stage) + '_' + str(block)
        else:
            path = path_fc + str(stage)
        w = file.get_tensor(path + '/weights')
        b = file.get_tensor(path + '/biases')
        return w, b

    def _conv2d(A_prev, W, strides=[1, 1], padding='SAME'):
        strides = [1, strides[0], strides[1], 1]
        return tf.nn.conv2d(A_prev, W, strides=strides, padding=padding)

    def conv_layer(A_prev, stage, block=None,
                   strides=[1, 1], padding='SAME',
                   freeze=True):
        w, b = _weights(stage, block)
        if freeze:
            w = tf.constant(w)
            b = tf.constant(b)
        else:
            w = tf.Variable(w)
            b = tf.Variable(b)
        c = _conv2d(A_prev, w, strides=strides, padding=padding)
        A = tf.nn.relu(tf.add(c, b), name='conv'+str(stage)+'_'+str(block))
        params = {'W': w, 'b': b}
        return A, params

    def fc_layer_wo_nonlin(A_prev, stage, is_final_layer=False, freeze=True):
        w, b = _weights(stage, type_code=1)
        if freeze:
            w = tf.constant(w)
            b = tf.constant(b)
        else:
            w = tf.Variable(w)
            b = tf.Variable(b)
        c = _conv2d(A_prev, w, padding='VALID')
        if is_final_layer:
            Z = tf.add(c, b, name='fc'+str(stage))
        else:
            Z = tf.add(c, b)
        params = {'W': w, 'b': b}
        return Z, params

    def fc_layer(A_prev, stage, freeze=True):
        Z, params = fc_layer_wo_nonlin(A_prev, stage, freeze=freeze)
        A = tf.nn.relu(Z, name='fc'+str(stage))
        params['Z'] = Z
        return A, params

    model = {}
    params = {}
    # max pool hyperparams
    KSIZE = [1, 2, 2, 1]
    STRIDES = [1, 2, 2, 1]
    PAD = 'VALID'

    if is_input_trainable:
        X = tf.get_variable(name='input', shape=input_shape)
    else:
        X = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')
    model['input'] = X

    # conv1_1
    model['conv1_1'], params['conv1_1'] = conv_layer(X, 1, block=1)

    # conv1_2
    model['conv1_2'], params['conv1_2'] = conv_layer(model['conv1_1'], 1, block=2)

    # pool 1
    model['pool_1'] = tf.nn.max_pool(model['conv1_2'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv2_1
    model['conv2_1'], params['conv2_1'] = conv_layer(model['pool_1'], 2, block=1)

    # conv2_2
    model['conv2_2'], params['conv2_2'] = conv_layer(model['conv2_1'], 2, block=2)

    # pool_2
    model['pool_2'] = tf.nn.max_pool(model['conv2_2'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv3_1
    model['conv3_1'], params['conv3_1'] = conv_layer(model['pool_2'], 3, block=1)

    # conv3_2
    model['conv3_2'], params['conv3_2'] = conv_layer(model['conv3_1'], 3, block=2)

    # conv3_3
    model['conv3_3'], params['conv3_3'] = conv_layer(model['conv3_2'], 3, block=3)

    # conv3_4
    model['conv3_4'], params['conv3_4'] = conv_layer(model['conv3_3'], 3, block=4)

    # pool_3
    model['pool_3'] = tf.nn.max_pool(model['conv3_4'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv4_1
    model['conv4_1'], params['conv4_1'] = conv_layer(model['pool_3'], 4, block=1)

    # conv4_2
    model['conv4_2'], params['conv4_2'] = conv_layer(model['conv4_1'], 4, block=2)

    # conv4_3
    model['conv4_3'], params['conv4_3'] = conv_layer(model['conv4_2'], 4, block=3)

    # conv4_4
    model['conv4_4'], params['conv4_4'] = conv_layer(model['conv4_3'], 4, block=4)

    # pool_4
    model['pool_4'] = tf.nn.max_pool(model['conv4_4'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv5_1
    model['conv5_1'], params['conv5_1'] = conv_layer(model['pool_4'], 5, block=1)

    # conv5_2
    model['conv5_2'], params['conv5_2'] = conv_layer(model['conv5_1'], 5, block=2)

    # conv5_3
    model['conv5_3'], params['conv5_3'] = conv_layer(model['conv5_2'], 5, block=3)

    # conv5_4
    model['conv5_4'], params['conv5_4'] = conv_layer(model['conv5_3'], 5, block=4)

    # pool_5
    model['pool_5'] = tf.nn.max_pool(model['conv5_4'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # fc6
    model['fc6'], params['fc6'] = fc_layer(model['pool_5'], 6)

    # fc7
    model['fc7'], params['fc6'] = fc_layer(model['fc6'], 7)

    # fc8
    if fine_tune_last:
        w = tf.get_variable('out_W', shape=[1, 1, 4096, n_classes])
        b = tf.get_variable('out_b', shape=[n_classes])
        model['out'] = tf.add(_conv2d(model['fc7'], w, padding='VALID'), b)
        params['out'] = {'W': w, 'b': b}
    else:
        model['out'], params['out'] = fc_layer_wo_nonlin(model['fc7'], 8)

    return model, params


if __name__ == '__main__':
    #model, params = vgg19(fine_tune_last=True, input_shape=[1, 224, 224, 3])
    model, params = vgg19(fine_tune_last=True, n_classes=10)
    pprint.pprint(model, indent=2)
    print()
    pprint.pprint(params, indent=2)