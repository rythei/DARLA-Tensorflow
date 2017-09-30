import numpy as np
import tensorflow as tf


def output_after_conv(input_size, kernel_size, padding, strides):
    W = input_size
    K = kernel_size
    S = strides
    if padding == 'SAME':
        P = (kernel_size-1)/2
    else:
        P = 0

    O = (W-K+2*P)/S + 1
    return int(O)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def add_noise(x, noise):
    return tf.add(x, noise)

