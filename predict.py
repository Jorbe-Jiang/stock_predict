from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

import tensorflow as tf

from configs import config_utils

from losses import model_losses
from model_utils import model_construction

from data_utils import data_loader

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

MODE_TRAIN = 'TRAIN'
MODE_VALIDATION = 'VALIDATION'
MODE_TEST = 'TEST'

tf.app.flags.DEFINE_enum(
    'mode', 'TRAIN', [MODE_TRAIN, MODE_VALIDATION, MODE_TEST],
    'What this binary will do.')

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            'Maximum number of steps to run.')

tf.app.flags.DEFINE_string(
    'attention_option', 'luong',
    "Attention mechanism.  One of [None, 'luong', 'bahdanau']")


tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    "Type of decoder optimizer.  One of ['sgd', 'adam']")

tf.app.flags.DEFINE_integer('print_every', 250,
                            'Frequency to print and log the '
                            'outputs of the model.')
tf.app.flags.DEFINE_integer('max_num_to_print', 5,
                            'Number of samples to log/print.')
tf.app.flags.DEFINE_integer('summaries_every', 100,
                            'Frequency to compute summaries.')

tf.app.flags.DEFINE_float('eval_interval_secs', 60,
                          'Delay for evaluating model.')

tf.app.flags.DEFINE_string('data_set', 'huangshan', 'Data set to operate on')
tf.app.flags.DEFINE_string('base_directory', './ckpts',
                           'Base directory for the logging, events and graph.')
tf.app.flags.DEFINE_string('data_dir', './data/huangshan',
                           'Directory for the training data.')
tf.app.flags.DEFINE_string('config_json', './configs/', 'config json file path.')
tf.app.flags.DEFINE_string('pretrained_ckpt', './ckpts/train',
                           'pretrained model to load.')


FLAGS = tf.app.flags.FLAGS


def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)


def load_model_configs(config_json):
    return config_utils.read_config_from_json_file(config_json)


def create_model(model_configs, is_training):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    new_learning_rate = tf.placeholder(tf.float32, [], name='new_learning_rate')
    learning_rate = tf.Variable(0.0, name='learning_rate', trainable=False)
    learning_rate_update = tf.assign(learning_rate, new_learning_rate)

    # Placeholders.
    history_prices = tf.placeholder(tf.float32, shape=[model_configs.batch_size, model_configs.time_seq_length, model_configs.num_price_types_used])
    month_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.time_seq_length])
    day_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.time_seq_length])
    weekday_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.time_seq_length])
    festival_periods_types_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.time_seq_length])
    dec_month_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.max_pred_days])
    dec_day_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.max_pred_days])
    dec_weekday_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.max_pred_days])
    dec_festival_periods_types_inputs = tf.placeholder(tf.int32, shape=[model_configs.batch_size, model_configs.max_pred_days])

    targets = tf.placeholder(tf.float32, shape=[model_configs.batch_size, model_configs.max_pred_days])
    targets_concat_prices = tf.placeholder(tf.float32, shape=[model_configs.batch_size, model_configs.max_pred_days,
                                           model_configs.num_price_types_used])

    inversed_scale_targets = inverse_scale_value(targets, model_configs.dataset_min_price_value, model_configs.dataset_max_price_value)

    if model_configs.use_time_emb:
        # [B, T], [B, T-1, 1]/[B, T-1, num_price_types_used]
        predictions, price_map_sequence = model_construction.create_graph(model_configs, is_training, history_prices,
                                                      month_inputs=month_inputs, day_inputs=day_inputs,
                                                      weekday_inputs=weekday_inputs,
                                                      festival_periods_types_inputs=festival_periods_types_inputs,
                                                      dec_month_inputs=dec_month_inputs,
                                                      dec_day_inputs=dec_day_inputs,
                                                      dec_weekday_inputs=dec_weekday_inputs,
                                                      dec_festival_periods_types_inputs=dec_festival_periods_types_inputs,
                                                      reuse=None)
    else:
        # [B, T], [B, T-1, 1]/[B, T-1, num_price_types_used]
        predictions, price_map_sequence = model_construction.create_graph(model_configs, is_training, history_prices, reuse=None)

    mse_loss = model_losses.mean_square_loss(predictions, inversed_scale_targets)

    # KL(softmax(predictions), softmax(targets)) and KL(softmax(targets), softmax(predictions))
    prob_pred = tf.nn.softmax(predictions, axis=1)
    prob_targ = tf.nn.softmax(inversed_scale_targets, axis=1)
    kl_pt_losses = tf.reduce_sum(prob_pred * (tf.log(prob_pred+1e-12) - tf.log(prob_targ+1e-12)), axis=1)
    kl_tp_losses = tf.reduce_sum(prob_targ * (tf.log(prob_targ + 1e-12) - tf.log(prob_pred + 1e-12)), axis=1)
    kl_pt_loss = tf.reduce_mean(kl_pt_losses)
    kl_tp_loss = tf.reduce_mean(kl_tp_losses)
    kl_loss = kl_pt_loss + 0.05 * kl_tp_loss
    price_emb_map_squared_mean_loss = tf.constant(0.0, dtype=tf.float32)
    if model_configs.use_price_emb_map_function and model_configs.max_pred_days > 1:
        if model_configs.only_use_pred_price_to_map:
            price_map_sequence = tf.squeeze(price_map_sequence, axis=-1)
            price_emb_map_squared_mean_loss = tf.losses.mean_squared_error(targets[:, :-1], price_map_sequence)
        else:
            price_emb_map_squared_mean_loss = tf.losses.mean_squared_error(targets_concat_prices[:, :-1, :], price_map_sequence)

    total_loss = mse_loss + model_configs.beta * price_emb_map_squared_mean_loss + model_configs.kl_penalty * kl_loss

    train_trainable_vars = tf.trainable_variables()

    train_op = None
    if is_training:
        train_gradients = tf.gradients(total_loss, train_trainable_vars)
        train_clipped_gradients, train_norm = tf.clip_by_global_norm(train_gradients, model_configs.grad_clipping)

        if FLAGS.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98)
        else:
            raise NotImplementedError

        print('Optimizing trainable vars.')
        for v in tf.trainable_variables():
            print(v)

        train_op = optimizer.apply_gradients(zip(train_clipped_gradients, train_trainable_vars), global_step=global_step)

        for v, g in zip(train_trainable_vars, train_gradients):
            variable_summaries(v, v.op.name)
            variable_summaries(g, 'grad/' + v.op.name)

    # Summaries.
    with tf.name_scope('general'):
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('generator_objectives'):
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('mse_loss', mse_loss)
        tf.summary.scalar('kl_loss', kl_loss)
        tf.summary.scalar('kl_pt_loss', kl_pt_loss)
        tf.summary.scalar('kl_tp_loss', kl_tp_loss)
        tf.summary.scalar('price_emb_map_squared_mean_loss', price_emb_map_squared_mean_loss)

    merge_summaries_op = tf.summary.merge_all()

    # Model saver.
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)

    Model = collections.namedtuple('Model', [
        'month_inputs', 'day_inputs', 'weekday_inputs', 'festival_periods_types_inputs', 'dec_month_inputs',
        'dec_day_inputs', 'dec_weekday_inputs', 'dec_festival_periods_types_inputs', 'history_prices',
        'targets', 'targets_concat_prices', 'total_loss', 'mse_loss', 'price_emb_map_squared_mean_loss', 'predictions',
        'price_map_sequence', 'train_op', 'merge_summaries_op', 'global_step',
        'new_learning_rate', 'learning_rate_update',
        'saver'
    ])

    model = Model(
        month_inputs, day_inputs, weekday_inputs, festival_periods_types_inputs, dec_month_inputs, dec_day_inputs,
        dec_weekday_inputs, dec_festival_periods_types_inputs, history_prices, targets, targets_concat_prices,
        total_loss, mse_loss,
        price_emb_map_squared_mean_loss, predictions, price_map_sequence,
        train_op, merge_summaries_op, global_step, new_learning_rate, learning_rate_update, saver)
    return model


def scale_value(value, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    scaled_value = scale * value + min - dataset_min_value * scale
    return scaled_value


def inverse_scale_value(scaled_value, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    value = (scaled_value + dataset_min_value * scale - min) / scale
    return value


def predict(model_configs, data, ckpts_dir):
    is_training = False

    plot_file = os.path.join(FLAGS.base_directory, 'predicts_test.png')
    gen_samples_file = os.path.join(FLAGS.base_directory, 'predictions_test.txt')
    log_wp = open(gen_samples_file, 'w')

    with tf.Graph().as_default():
        tf.set_random_seed(2019)
        container_name = ''
        with tf.container(container_name):
            # Construct the model.
            model = create_model(model_configs, is_training)

            trainable_variables = tf.trainable_variables()
            trainable_variables.append(model.global_step)
            eval_saver = tf.train.Saver(var_list=trainable_variables)
            sv = tf.train.Supervisor()
            sess = sv.PrepareSession('', start_standard_services=False)

            model_save_path = tf.train.latest_checkpoint(ckpts_dir)
            if not model_save_path:
                tf.logging.warning('No checkpoint yet in: %s', ckpts_dir)
                return

            tf.logging.info('Starting eval of: %s' % model_save_path)
            tf.logging.info('Only restoring trainable variables.')
            eval_saver.restore(sess, model_save_path)

            np.random.seed(2019)

            (history_prices, month_inputs, day_inputs, weekday_inputs, festival_periods_types_inputs, dec_month_inputs, dec_day_inputs, dec_weekday_inputs, dec_festival_periods_types_inputs, year_inputs, dec_year_inputs) = data

            eval_feed = {
                model.history_prices: history_prices,
                model.month_inputs: month_inputs,
                model.day_inputs: day_inputs,
                model.weekday_inputs: weekday_inputs,
                model.festival_periods_types_inputs: festival_periods_types_inputs,
                model.dec_month_inputs: dec_month_inputs,
                model.dec_day_inputs: dec_day_inputs,
                model.dec_weekday_inputs: dec_weekday_inputs,
                model.dec_festival_periods_types_inputs: dec_festival_periods_types_inputs
            }

            [predictions, step] = sess.run(
                [model.predictions,
                model.global_step], feed_dict=eval_feed)

            print('global_step: %d' % step)

            history_prices = history_prices[:, :, 0].tolist()[0]
            month_inputs = month_inputs.tolist()[0]
            day_inputs = day_inputs.tolist()[0]
            dec_month_inputs = dec_month_inputs.tolist()[0]
            dec_day_inputs = dec_day_inputs.tolist()[0]
            year_inputs = year_inputs.tolist()[0]
            dec_year_inputs = dec_year_inputs.tolist()[0]

            predictions = predictions.tolist()[0]

            history_data = pd.DataFrame(index=range(0, len(history_prices)), columns=['Date', 'Prices'])

            for i in range(len(history_prices)):
                year = year_inputs[i]
                month = month_inputs[i]+1
                day = day_inputs[i]+1
                price = history_prices[i]
                inverse_scaled_price = inverse_scale_value(price, model_configs.dataset_min_price_value,
                                                           model_configs.dataset_max_price_value,
                                                           min=0.0, max=1.0)
                history_data['Date'][i] = pd.Timestamp(year, month, day)
                history_data['Prices'][i] = inverse_scaled_price

            history_data['Date'] = pd.to_datetime(history_data.Date, format='%Y-%m-%d')
            history_data.index = history_data['Date']

            pred_data = pd.DataFrame(index=range(0, len(predictions)), columns=['Date', 'Preds'])

            for i in range(len(predictions)):
                year = dec_year_inputs[i]
                month = dec_month_inputs[i]+1
                day = dec_day_inputs[i]+1
                pred_price = predictions[i]
                inverse_scaled_pred_price = pred_price

                pred_data['Date'][i] = pd.Timestamp(year, month, day)
                pred_data['Preds'][i] = inverse_scaled_pred_price

            pred_data['Date'] = pd.to_datetime(pred_data.Date, format='%Y-%m-%d')
            pred_data.index = pred_data['Date']

            plt.figure(figsize=(16, 8))
            plt.plot(history_data['Prices'])
            plt.plot(pred_data['Preds'], label='Close Price history')
            plt.savefig(plot_file)

            for i, prediction in enumerate(predictions):
                date = pred_data['Date'][i]
                inverse_scaled_prediction = prediction
                log_wp.write('Year: %d Month: %d Day: %d ,prediction: %.5f \n' % (date.year, date.month, date.day, inverse_scaled_prediction))

        tf.logging.error('Done.')
        log_wp.close()


def main(_):
    model_configs = load_model_configs(FLAGS.config_json)
    data_set, dataset_price_min_value, dataset_price_max_value = data_loader.load_test_raw_data(model_configs, FLAGS.data_dir)
    model_configs.dataset_price_min_value = dataset_price_min_value
    model_configs.dataset_price_max_value = dataset_price_max_value

    tf.gfile.MakeDirs(FLAGS.base_directory)

    predict(model_configs, data_set, FLAGS.pretrained_ckpt)


if __name__ == '__main__':
    tf.app.run()
