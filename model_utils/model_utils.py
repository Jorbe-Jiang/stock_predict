from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

np.random.seed(2019)

FLAGS = tf.app.flags.FLAGS


def assign_learning_rate(session, lr_update, lr_placeholder, new_lr):
    session.run(lr_update, feed_dict={lr_placeholder: new_lr})


def clip_weights(variables, c_lower, c_upper):
    clip_ops = []

    for var in variables:
        clipped_var = tf.clip_by_value(var, c_lower, c_upper)
        clip_ops.append(tf.assign(var, clipped_var))
    return tf.group(*clip_ops)


def retrieve_init_savers(model_configs):
    init_savers = {}
    trainable_vars = tf.trainable_variables()

    saver = tf.train.Saver(var_list=trainable_vars)
    init_savers['init_saver'] = saver

    return init_savers


def init_fn(init_savers, sess):
    if FLAGS.pretrained_ckpt:
        model_ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_ckpt)
        if model_ckpt and tf.train.checkpoint_exists(model_ckpt.model_checkpoint_path):
            print('Restoring pretrained model from %s.' % model_ckpt.model_checkpoint_path)
            tf.logging.info('Restoring pretrained model from %s.' % model_ckpt.model_checkpoint_path)
            init_savers['init_saver'].restore(sess, model_ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found in {}!!!'.format(FLAGS.pretrained_ckpt))
            assert ('No checkpoint found in {}!!!'.format(FLAGS.pretrained_ckpt))
            exit()
    else:
        assert 'pretrained_ckpt is NONE!!!'

