from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from models import encoder
from models import decoder

FLAGS = tf.app.flags.FLAGS


def create_graph(model_configs,
                 is_training,
                 history_prices,
                 month_inputs=None,
                 day_inputs=None,
                 weekday_inputs=None,
                 festival_periods_types_inputs=None,
                 dec_month_inputs=None,
                 dec_day_inputs=None,
                 dec_weekday_inputs=None,
                 dec_festival_periods_types_inputs=None,
                 reuse=None):
    if FLAGS.attention_option == '':
        FLAGS.attention_option = None

    month_embedding = None
    day_embedding = None
    weekday_embedding = None
    festival_periods_types_embedding = None
    if model_configs.use_time_emb:
        assert month_inputs is not None
        assert day_inputs is not None
        with tf.variable_scope('encoder_month_embedding', reuse=reuse):
            month_embedding = tf.get_variable('month_embedding', [model_configs.num_months, model_configs.month_emb_size])

        with tf.variable_scope('encoder_day_embedding', reuse=reuse):
            day_embedding = tf.get_variable('day_embedding', [model_configs.num_days, model_configs.day_emb_size])

        if model_configs.use_weekday_emb:
            assert weekday_inputs is not None
            with tf.variable_scope('encoder_weekday_embedding', reuse=reuse):
                weekday_embedding = tf.get_variable('weekday_embedding', [model_configs.num_weekdays, model_configs.weekday_emb_size])

        if model_configs.use_festival_periods_emb:
            assert festival_periods_types_inputs is not None
            with tf.variable_scope('encoder_festival_periods_types_embedding', reuse=reuse):
                festival_periods_types_embedding = tf.get_variable('festival_periods_types_embedding', [model_configs.num_festival_periods_types, model_configs.festival_periods_emb_size])

    embedding_inputs = None
    if model_configs.use_time_emb:
        month_emb_inputs = tf.nn.embedding_lookup(month_embedding, month_inputs)
        day_emb_inputs = tf.nn.embedding_lookup(day_embedding, day_inputs)

        # [B, T, month_emb_size+day_emb_size]
        embedding_inputs = tf.concat([month_emb_inputs, day_emb_inputs], axis=2)
        if model_configs.use_weekday_emb:
            weekday_emb_inputs = tf.nn.embedding_lookup(weekday_embedding, weekday_inputs)
            embedding_inputs = tf.concat([embedding_inputs, weekday_emb_inputs], axis=2)

        if model_configs.use_festival_periods_emb:
            festival_periods_types_emb_inputs = tf.nn.embedding_lookup(festival_periods_types_embedding,
                                                                       festival_periods_types_inputs)
            embedding_inputs = tf.concat([embedding_inputs, festival_periods_types_emb_inputs], axis=2)

        # transform_history_prices = tf.layers.dense(tf.tanh(history_prices), model_configs.transform_price_feature_dim,
        #                                            activation=tf.nn.relu,
        #                                            name='enc_transform_price_feature', reuse=tf.AUTO_REUSE)
        # embedding_inputs = tf.concat([transform_history_prices, embedding_inputs], axis=2)
        # embedding_inputs = tf.concat([tf.tanh(history_prices), embedding_inputs], axis=2)
    else:
        # transform_history_prices = tf.layers.dense(tf.tanh(history_prices), model_configs.transform_price_feature_dim,
        #                                            activation=tf.nn.relu, name='enc_transform_price_feature',
        #                                            reuse=tf.AUTO_REUSE)

        # [B, T, transform_price_feature_dim]
        # embedding_inputs = transform_history_prices
        # embedding_inputs = tf.tanh(history_prices)
        embedding_inputs = history_prices

    seq_length = tf.constant(model_configs.time_seq_length, dtype=tf.int32, shape=[model_configs.batch_size])
    enc_outputs, enc_states = encoder.encoder(embedding_inputs, seq_length, model_configs,
                                              is_training=is_training, bert_config=None,
                                              input_ids=None, input_mask=None,
                                              name='encoder', reuse=reuse)
    price_map_sequence = []
    if model_configs.max_pred_days == 1:
        if model_configs.enc_num_layers == 1 and model_configs.encoder_type == 'birnn':
            final_state = enc_states.h
        else:
            final_state = enc_states[-1].h

        preds = tf.layers.dense(final_state, 1, activation=None, name='logits_dense', reuse=tf.AUTO_REUSE)

        predictions_sequence = tf.reshape(preds, [-1, 1])
        return predictions_sequence, price_map_sequence

    # [B, T], [B, T-1, 1]
    predictions_sequence, price_map_sequence = decoder.lstm_decoder(model_configs, is_training, enc_outputs, enc_states, history_prices,
                                                                    dec_month_inputs=dec_month_inputs,
                                                                    dec_day_inputs=dec_day_inputs,
                                                                    dec_weekday_inputs=dec_weekday_inputs,
                                                                    dec_festival_periods_types_inputs=dec_festival_periods_types_inputs,
                                                                    month_embedding=month_embedding,
                                                                    day_embedding=day_embedding,
                                                                    weekday_embedding=weekday_embedding,
                                                                    festival_periods_types_embedding=festival_periods_types_embedding,
                                                                    attention_option=FLAGS.attention_option,
                                                                    name='lstm_decoder', reuse=None)

    return predictions_sequence, price_map_sequence
