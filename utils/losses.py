import sys; sys.path.append('..')
import keras.backend as K
import tensorflow as tf


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def categorical_crossentropy(y, p):
    return K.mean(K.categorical_crossentropy(y, p))


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def make_loss(loss_name):
    if loss_name == 'bce':
        def loss(y, p):
            return binary_crossentropy(y, p)
        return loss
    elif loss_name == 'focal_loss':
        def loss(y, p):
            return focal_loss(y, p)
        return loss
    elif loss_name == 'cce':
        def loss(y, p):
            return categorical_crossentropy(y, p)
        return loss
    else:
        ValueError("Unknown loss.")
