from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mean_square_loss(predictions, targets):
    loss = tf.losses.mean_squared_error(targets, predictions)
    return loss

