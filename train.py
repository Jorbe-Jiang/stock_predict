from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
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
from model_utils import model_utils
from model_utils import evaluation_utils

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


def get_iterator(model_configs, raw_data):
    """Return the data iterator."""
    iterator = data_loader.data_iterator(model_configs, raw_data)
    return iterator


def train_model(model_configs, data, log_dir, log):
    is_training = True

    with tf.Graph().as_default():
        tf.set_random_seed(2019)
        # Construct the model.
        model = create_model(model_configs, is_training)

        print('\nTrainable Variables in Graph:')
        for v in tf.trainable_variables():
            print(v)

        init_savers = model_utils.retrieve_init_savers(model_configs)
        init_fn = partial(model_utils.init_fn, init_savers)

        sv = tf.train.Supervisor(
            logdir=log_dir,
            is_chief=True,
            saver=model.saver,
            global_step=model.global_step,
            save_model_secs=60,
            recovery_wait_secs=30,
            summary_op=None,
            init_fn=init_fn)

        with sv.managed_session('') as sess:
            step = 1
            epoch = 0

            while epoch < model_configs.num_epochs:
                epoch += 1
                avg_epoch_loss = []
                iterator = get_iterator(model_configs, data)

                for (history_prices, month_inputs, day_inputs, weekday_inputs, festival_periods_types_inputs, dec_month_inputs, dec_day_inputs, dec_weekday_inputs, dec_festival_periods_types_inputs, targets, year_inputs, dec_year_inputs, targets_concat_prices) in iterator:
                    train_feed = {
                        model.history_prices: history_prices,
                        model.month_inputs: month_inputs,
                        model.day_inputs: day_inputs,
                        model.weekday_inputs: weekday_inputs,
                        model.festival_periods_types_inputs: festival_periods_types_inputs,
                        model.dec_month_inputs: dec_month_inputs,
                        model.dec_day_inputs: dec_day_inputs,
                        model.dec_weekday_inputs: dec_weekday_inputs,
                        model.dec_festival_periods_types_inputs: dec_festival_periods_types_inputs,
                        model.targets: targets,
                        model.targets_concat_prices: targets_concat_prices
                    }

                    lr = (model_configs.lr - model_configs.max_lr) * ((1 - step / model_configs.warm_up_steps) ** (2.0)) + model_configs.max_lr
                    lr = max(lr, model_configs.min_lr)
                    gen_learning_rate = lr
                    model_utils.assign_learning_rate(sess, model.learning_rate_update, model.new_learning_rate, gen_learning_rate)
                    [_, loss, step] = sess.run(
                        [
                            model.train_op, model.total_loss,
                            model.global_step
                        ],
                        feed_dict=train_feed)

                    if step % 20 == 0:
                        model.saver.save(sess, os.path.join(FLAGS.base_directory, 'train/my-model'), global_step=step)

                    avg_epoch_loss.append(loss)

                    print('Epoch: %d  global_step: %d' % (epoch, step))
                    print('total_loss: %.3f' % loss)

                    if step % FLAGS.summaries_every == 0:
                        summary_str = sess.run(model.merge_summaries_op, feed_dict=train_feed)
                        sv.SummaryComputed(sess, summary_str)

                    if step % FLAGS.print_every == 0:
                        print('Epoch: %d  global_step: %d' % (epoch, step))
                        print(' learning_rate: %.6f' % gen_learning_rate)
                        log.write('Epoch: %d  global_step: %d\n' % (epoch, step))
                        log.write(' learning_rate: %.6f\n' % gen_learning_rate)

                        evaluation_utils.print_and_log_losses(log, step, avg_epoch_loss)
                        evaluation_utils.generate_logs(model_configs, sess, model, log, train_feed)
                        log.flush()
    log.close()


def evaluate_model(model_configs, data, train_dir, log):
    is_training = False

    if FLAGS.mode == MODE_VALIDATION:
        logdir = FLAGS.base_directory + '/validation'
    elif FLAGS.mode == MODE_TEST:
        logdir = FLAGS.base_directory + '/test'
    else:
        raise NotImplementedError

    tf.gfile.MakeDirs(os.path.join(FLAGS.base_directory, 'plots'))
    gen_samples_file = os.path.join(FLAGS.base_directory, 'predictions.txt')
    log_wp = open(gen_samples_file, 'w')

    with tf.Graph().as_default():
        tf.set_random_seed(2019)
        container_name = ''
        with tf.container(container_name):
            # Construct the model.
            model = create_model(model_configs, is_training)
            print('\nTrainable Variables in Graph:')
            for v in tf.trainable_variables():
                print(v)

            trainable_variables = tf.trainable_variables()
            trainable_variables.append(model.global_step)
            eval_saver = tf.train.Saver(var_list=trainable_variables)
            sv = tf.train.Supervisor(logdir=logdir)
            sess = sv.PrepareSession('', start_standard_services=False)

            tf.logging.info('Evaluating model.')
            model_save_path = tf.train.latest_checkpoint(train_dir)
            if not model_save_path:
                tf.logging.warning('No checkpoint yet in: %s', train_dir)
                return

            tf.logging.info('Starting eval of: %s' % model_save_path)
            tf.logging.info('Only restoring trainable variables.')
            eval_saver.restore(sess, model_save_path)

            avg_epoch_loss = []
            np.random.seed(2019)

            iterator = get_iterator(model_configs, data)

            idx = 0

            for (history_prices, month_inputs, day_inputs, weekday_inputs, festival_periods_types_inputs, dec_month_inputs,
                 dec_day_inputs, dec_weekday_inputs, dec_festival_periods_types_inputs, targets, year_inputs, dec_year_inputs,
                 targets_concat_prices) in iterator:
                eval_feed = {
                    model.history_prices: history_prices,
                    model.month_inputs: month_inputs,
                    model.day_inputs: day_inputs,
                    model.weekday_inputs: weekday_inputs,
                    model.festival_periods_types_inputs: festival_periods_types_inputs,
                    model.dec_month_inputs: dec_month_inputs,
                    model.dec_day_inputs: dec_day_inputs,
                    model.dec_weekday_inputs: dec_weekday_inputs,
                    model.dec_festival_periods_types_inputs: dec_festival_periods_types_inputs,
                    model.targets: targets,
                    model.targets_concat_prices: targets_concat_prices
                }

                [loss, predictions, price_map_sequence, step] = sess.run(
                    [model.mse_loss, model.predictions, model.price_map_sequence,
                    model.global_step], feed_dict=eval_feed)

                print('global_step: %d' % step)
                print('mse_loss: %.3f' % loss)
                idx += 1

                targets = targets.tolist()[0]
                predictions = predictions.tolist()[0]
                price_map_sequence = price_map_sequence.tolist()[0]
                if idx % FLAGS.print_every == 0:
                    assert model_configs.batch_size == 1

                    plot_file = os.path.join(os.path.join(FLAGS.base_directory, 'plots'), 'predictions_{}.png'.format(idx))

                    history_prices = history_prices[:, :, 0].tolist()[0]
                    month_inputs = month_inputs.tolist()[0]
                    day_inputs = day_inputs.tolist()[0]
                    dec_month_inputs = dec_month_inputs.tolist()[0]
                    dec_day_inputs = dec_day_inputs.tolist()[0]
                    year_inputs = year_inputs.tolist()[0]
                    dec_year_inputs = dec_year_inputs.tolist()[0]

                    history_data = pd.DataFrame(index=range(0, len(history_prices)), columns=['Date', 'Prices'])

                    for i in range(len(history_prices)):
                        year = year_inputs[i]
                        month = month_inputs[i]+1
                        day = day_inputs[i]+1
                        price = history_prices[i]
                        inverse_scaled_price = inverse_scale_value(price, model_configs.dataset_min_price_value,
                                                                   model_configs.dataset_max_price_value, min=0.0, max=1.0)
                        history_data['Date'][i] = pd.Timestamp(year, month, day)
                        history_data['Prices'][i] = inverse_scaled_price

                    history_data['Date'] = pd.to_datetime(history_data.Date, format='%Y-%m-%d')
                    history_data.index = history_data['Date']

                    pred_data = pd.DataFrame(index=range(0, len(predictions)), columns=['Date', 'Prices', 'Preds', 'Map_Preds'])

                    for i in range(len(predictions)):
                        year = dec_year_inputs[i]
                        month = dec_month_inputs[i]+1
                        day = dec_day_inputs[i]+1
                        pred_price = predictions[i]
                        true_price = targets[i]
                        inverse_scaled_pred_price = pred_price
                        inverse_scaled_true_price = inverse_scale_value(true_price, model_configs.dataset_min_price_value,
                                                                        model_configs.dataset_max_price_value, min=0.0, max=1.0)
                        pred_data['Date'][i] = pd.Timestamp(year, month, day)
                        pred_data['Prices'][i] = inverse_scaled_true_price
                        pred_data['Preds'][i] = inverse_scaled_pred_price

                    for i in range(len(price_map_sequence)):
                        if model_configs.only_use_pred_price_to_map:
                            map_pred_price = price_map_sequence[i]
                        else:
                            map_pred_price = price_map_sequence[i][0]
                        inverse_scaled_map_pred_price = inverse_scale_value(map_pred_price,
                                                                            model_configs.dataset_min_price_value,
                                                                            model_configs.dataset_max_price_value,
                                                                            min=0.0, max=1.0)
                        pred_data['Map_Preds'][i] = inverse_scaled_map_pred_price

                    pred_data['Map_Preds'][len(predictions)-1] = pred_data['Map_Preds'][len(price_map_sequence)-1]  # 确保长度和Preds一样
                    pred_data['Date'] = pd.to_datetime(pred_data.Date, format='%Y-%m-%d')
                    pred_data.index = pred_data['Date']

                    plt.figure(figsize=(16, 8))
                    plt.plot(history_data['Prices'])
                    # plt.plot(pred_data[['Preds', 'Prices', 'Map_Preds']], label='Close Price history')
                    plt.plot(pred_data[['Prices', 'Preds']], label='Close Price history')
                    plt.savefig(plot_file)

                for j, (prediction, target) in enumerate(zip(predictions, targets)):
                    if j < len(predictions) - 1:
                        if model_configs.only_use_pred_price_to_map:
                            map_pred_price = price_map_sequence[j]
                        else:
                            map_pred_price = price_map_sequence[j][0]
                        inverse_scaled_map_pred_price = inverse_scale_value(map_pred_price, model_configs.dataset_min_price_value,
                                                                    model_configs.dataset_max_price_value, min=0.0,
                                                                    max=1.0)
                    else:
                        inverse_scaled_map_pred_price = 0.0

                    inverse_scaled_prediction = prediction
                    inverse_scaled_target = inverse_scale_value(target, model_configs.dataset_min_price_value,
                                                                model_configs.dataset_max_price_value,
                                                                min=0.0, max=1.0)

                    log_wp.write('prediction: %.5f, map_prediction: %.5f \n' % (inverse_scaled_prediction, inverse_scaled_map_pred_price))
                    log_wp.write('target: %.5f \n' % inverse_scaled_target)
                log_wp.write('-'*100+'\n')

                avg_epoch_loss.append(loss)
                evaluation_utils.generate_logs(model_configs, sess, model, log, eval_feed)

            evaluation_utils.print_and_log_losses(log, step, avg_epoch_loss)
            summary_str = sess.run(model.merge_summaries_op, feed_dict=eval_feed)
            sv.SummaryComputed(sess, summary_str)

        tf.logging.error('Done.')
        log_wp.close()


def main(_):
    model_configs = load_model_configs(FLAGS.config_json)

    train_dir = FLAGS.base_directory + '/train'

    if FLAGS.mode == MODE_TRAIN:
        train_data_set, valid_data_set, dataset_price_min_value, dataset_price_max_value = data_loader.load_raw_data(model_configs, FLAGS.data_dir)
    elif FLAGS.mode == MODE_VALIDATION:
        train_data_set, valid_data_set, dataset_price_min_value, dataset_price_max_value = data_loader.load_raw_data(model_configs, FLAGS.data_dir)
    else:
        raise NotImplementedError

    # reset price min max value
    model_configs.dataset_min_price_value = dataset_price_min_value
    model_configs.dataset_max_price_value = dataset_price_max_value

    tf.gfile.MakeDirs(FLAGS.base_directory)

    if FLAGS.mode == MODE_TRAIN:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, FLAGS.data_set+'_train-log.txt'), mode='w')
    elif FLAGS.mode == MODE_VALIDATION:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, FLAGS.data_set+'_validation-log.txt'), mode='w')
    else:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, FLAGS.data_set+'_test-log.txt'), mode='w')

    if FLAGS.mode == MODE_TRAIN:
        train_model(model_configs, train_data_set, train_dir, log)
    elif FLAGS.mode == MODE_VALIDATION:
        evaluate_model(model_configs, valid_data_set, train_dir, log)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    tf.app.run()
