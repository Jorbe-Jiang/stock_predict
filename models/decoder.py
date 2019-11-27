from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from module_op_utils import attention_utils


def scale_value(fake_price, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    """
    对预测价格进行scale处理, 处理方式和数据预处理过程中的股票价格scale方式一致
    """
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    scaled_value = scale * fake_price + min - dataset_min_value * scale
    return scaled_value


def lstm_decoder(model_configs,
                 is_training,
                 encoder_outputs,
                 encoder_states,
                 history_prices,
                 dec_month_inputs=None,
                 dec_day_inputs=None,
                 dec_weekday_inputs=None,
                 dec_festival_periods_types_inputs=None,
                 month_embedding=None,
                 day_embedding=None,
                 weekday_embedding=None,
                 festival_periods_types_embedding=None,
                 attention_option='luong',
                 name='lstm_decoder',
                 reuse=None):
    if model_configs.encoder_type == 'birnn':
        dec_hidden_size = model_configs.enc_hidden_size * 2
    elif model_configs.encoder_type == 'cnn':
        dec_hidden_size = model_configs.enc_num_filters
    else:
        dec_hidden_size = model_configs.enc_hidden_size

    with tf.variable_scope(name, reuse=reuse):
        def dec_lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(dec_hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        dec_attn_cell = dec_lstm_cell

        if is_training and model_configs.dec_keep_prob < 1:
            def dec_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(dec_lstm_cell(), output_keep_prob=model_configs.dec_keep_prob)

        if model_configs.enc_num_layers > 1:
            cell_dec = tf.contrib.rnn.MultiRNNCell([dec_attn_cell() for _ in range(model_configs.enc_num_layers)],
                                                   state_is_tuple=True)
        else:
            cell_dec = dec_attn_cell()

        state_dec = encoder_states

        if attention_option is not None:
            with tf.variable_scope('dec_attention', reuse=reuse):
                (dec_attention_keys, dec_attention_values, dec_attention_score_fn,
                 dec_attention_construct_fn) = attention_utils.prepare_attention(encoder_outputs,
                                                                                 attention_option,
                                                                                 num_units=dec_hidden_size, reuse=reuse)

        def make_mask(keep_prob, units):
            random_tensor = keep_prob
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            random_tensor += tf.random_uniform(tf.stack([model_configs.batch_size, units]))
            return tf.floor(random_tensor) / keep_prob

        if is_training:
            dec_output_mask = make_mask(model_configs.dec_keep_prob, dec_hidden_size)

        dec_time_emb_inputs = None
        if model_configs.use_time_emb:
            dec_month_emb_inputs = tf.nn.embedding_lookup(month_embedding, dec_month_inputs)
            dec_day_emb_inputs = tf.nn.embedding_lookup(day_embedding, dec_day_inputs)
            # [B, T, month_emb_size+day_emb_size]
            dec_time_emb_inputs = tf.concat([dec_month_emb_inputs, dec_day_emb_inputs], axis=2)
            if model_configs.use_weekday_emb:
                weekday_emb_inputs = tf.nn.embedding_lookup(weekday_embedding, dec_weekday_inputs)
                dec_time_emb_inputs = tf.concat([dec_time_emb_inputs, weekday_emb_inputs], axis=2)
            else:
                assert weekday_embedding is None

            if model_configs.use_festival_periods_emb:
                festival_periods_types_emb_inputs = tf.nn.embedding_lookup(festival_periods_types_embedding,
                                                                           dec_festival_periods_types_inputs)
                dec_time_emb_inputs = tf.concat([dec_time_emb_inputs, festival_periods_types_emb_inputs], axis=2)
            else:
                assert festival_periods_types_embedding is None

        else:
            assert month_embedding is None
            assert day_embedding is None

        preds_sequence = []
        price_map_sequence = []

        # if model_configs.only_use_pred_price_to_map:
        #     dec_srt_emb = tf.get_variable('dec_srt_emb', [1, 1])
        #     tiled_dec_srt_emb = tf.tile(dec_srt_emb, [model_configs.batch_size, 1])
        #     tiled_dec_srt_emb = tf.reshape(tiled_dec_srt_emb, [model_configs.batch_size, 1])
        # else:
        #     dec_srt_emb = tf.get_variable('dec_srt_emb', [1, model_configs.num_price_types_used])
        #     tiled_dec_srt_emb = tf.tile(dec_srt_emb, [model_configs.batch_size, 1])
        #     tiled_dec_srt_emb = tf.reshape(tiled_dec_srt_emb, [model_configs.batch_size, model_configs.num_price_types_used])

        if model_configs.use_price_emb_map_function:
            if model_configs.only_use_pred_price_to_map:
                price_emb_map_func_w_1 = tf.get_variable('price_emb_map_func_w_1', [1, 3])
                price_emb_map_func_b_1 = tf.get_variable('price_emb_map_func_b_1', [3])
                price_emb_map_func_w = tf.get_variable('price_emb_map_func_w', [3, 1])
                price_emb_map_func_b = tf.get_variable('price_emb_map_func_b', [1])
            else:
                price_emb_map_func_w_1 = tf.get_variable('price_emb_map_func_w_1', [1, model_configs.num_price_types_used])
                price_emb_map_func_b_1 = tf.get_variable('price_emb_map_func_b_1', [model_configs.num_price_types_used])
                price_emb_map_func_w = tf.get_variable('price_emb_map_func_w', [model_configs.num_price_types_used, model_configs.num_price_types_used])
                price_emb_map_func_b = tf.get_variable('price_emb_map_func_b', [model_configs.num_price_types_used])

        fake_price = None
        for t in range(model_configs.max_pred_days):
            if t > 0:
                tf.get_variable_scope().reuse_variables()

            if t == 0:
                # history_price = tiled_dec_srt_emb
                if model_configs.only_use_pred_price_to_map:
                    # 预测目标价格信息在history_prices最后一个维度的第1维
                    history_price = tf.reshape(history_prices[:, -1, 0], [-1, 1])
                else:
                    history_price = tf.reshape(history_prices[:, -1, :], [-1, model_configs.num_price_types_used])
            else:
                scaled_fake_price = scale_value(fake_price, model_configs.dataset_min_price_value, model_configs.dataset_max_price_value)
                history_price = tf.reshape(scaled_fake_price, [-1, 1])
                # history_price = tf.reshape(fake_price, [-1, 1])
                if model_configs.use_price_emb_map_function:
                    # 线性price embedding映射函数, [B, 1]/[B, num_price_types_used]
                    history_price = tf.nn.bias_add(tf.matmul(history_price, price_emb_map_func_w_1), price_emb_map_func_b_1)
                    history_price = tf.nn.relu(history_price)
                    history_price = tf.nn.bias_add(tf.matmul(history_price, price_emb_map_func_w), price_emb_map_func_b)

                    price_map_sequence.append(history_price)

            dec_rnn_inp = history_price
            dec_rnn_inp = tf.tanh(dec_rnn_inp)
            # # 将price特征信息映射到更高的维度
            # dec_rnn_inp = tf.layers.dense(dec_rnn_inp, model_configs.transform_price_feature_dim, activation=tf.nn.relu, name='dec_transform_price_feature', reuse=tf.AUTO_REUSE)

            if model_configs.use_time_emb:
                dec_time_emb_input = dec_time_emb_inputs[:, t]
                dec_rnn_inp = tf.concat([dec_rnn_inp, dec_time_emb_input], axis=1)

            with tf.variable_scope('dec_rnn'):
                dec_rnn_out, state_dec = cell_dec(dec_rnn_inp, state_dec)

                if attention_option is not None:
                    dec_rnn_out = dec_attention_construct_fn(dec_rnn_out, dec_attention_keys, dec_attention_values)

            if is_training:
                dec_rnn_out *= dec_output_mask

            preds = tf.layers.dense(dec_rnn_out, 1, activation=None, name='logits_dense', reuse=tf.AUTO_REUSE)

            preds = tf.reshape(preds, [-1])
            fake_price = preds
            preds_sequence.append(preds)

    # [B, T]
    preds_sequence = tf.stack(preds_sequence, axis=1)
    if model_configs.use_price_emb_map_function:
        # [B, T-1, 1]
        price_map_sequence = tf.stack(price_map_sequence, axis=1)

    return preds_sequence, price_map_sequence
