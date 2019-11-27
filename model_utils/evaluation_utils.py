from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS


def inverse_scale_value(scaled_value, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    value = (scaled_value + dataset_min_value * scale - min) / scale
    return np.round(value, decimals=3)


def print_and_log_losses(log, step, avg_epoch_loss):
    print('global_step: %d' % step)
    print(' loss(avg epoch): %.5f' % np.mean(avg_epoch_loss))
    log.write('\nglobal_step: %d\n' % step)
    log.write(' loss(avg epoch): %.5f\n' % np.mean(avg_epoch_loss))


def print_and_log(model_configs, log, predictions, targets, max_num_to_print=5):
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        inverse_scaled_prediction = prediction
        inverse_scaled_target = inverse_scale_value(target, model_configs.dataset_min_price_value,
                                                    model_configs.dataset_max_price_value, min=0.0, max=1.0)
        if i < max_num_to_print:
            print('Prediction', i, '. ', str(inverse_scaled_prediction))
            print('Target', i, '.   ', str(inverse_scaled_target))
        log.write('\nPrediction ' + str(i) + '. ' + ' '.join([str(value) for value in inverse_scaled_prediction]))
        log.write('\nTarget ' + str(i) + '.   ' + ' '.join([str(value) for value in inverse_scaled_target]))
    log.write('\n')
    print('\n')
    log.flush()


def generate_logs(model_configs, sess, model, log, feed):
    [
        predictions, mse_loss
    ] = sess.run(
          [
                model.predictions,
                model.mse_loss
          ],
          feed_dict=feed)

    targets = feed[model.targets]

    print('Predictions, MSE_Loss: {}'.format(np.round(mse_loss, decimals=3)))
    log.write('Predictions, MSE_Loss: {}\n'.format(np.round(mse_loss, decimals=3)))
    print_and_log(model_configs, log, predictions, targets, FLAGS.max_num_to_print)
