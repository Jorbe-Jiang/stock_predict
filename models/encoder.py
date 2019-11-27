from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from module_op_utils import encoder_modules


def encoder(embedding_inputs, inputs_sequence_length, model_configs, is_training=True, bert_config=None, input_ids=None,
                      input_mask=None, name='baseline_encoder', reuse=None):
    if is_training:
        keep_prob = model_configs.enc_keep_prob
    else:
        keep_prob = 1.0

    if model_configs.encoder_type == 'rnn':
        outputs, states = encoder_modules.rnn_encoder(embedding_inputs, inputs_sequence_length,
                                                      model_configs.enc_hidden_size, model_configs.enc_num_layers,
                                                      model_configs.batch_size, is_training=is_training,
                                                      keep_prob=keep_prob, rnn_type=model_configs.rnn_type,
                                                      birnn=False, name=name + '_rnn', reuse=reuse)
        return outputs, states
    elif model_configs.encoder_type == 'birnn':
        outputs, states = encoder_modules.rnn_encoder(embedding_inputs, inputs_sequence_length,
                                                      model_configs.enc_hidden_size, model_configs.enc_num_layers,
                                                      model_configs.batch_size, is_training=is_training,
                                                      keep_prob=keep_prob, rnn_type=model_configs.rnn_type,
                                                      birnn=True, name=name + '_birnn', reuse=reuse)
        return outputs, states
    elif model_configs.encoder_type == 'cnn':
        assert model_configs.max_pooling == 1
        emb_size = tf.shape(embedding_inputs)[-1]
        conv_outputs = encoder_modules.cnn_encoder(embedding_inputs, emb_size, model_configs.enc_num_filters,
                                                   filter_sizes=model_configs.enc_filter_sizes,
                                                   max_pooling=model_configs.max_pooling, name=name + '_cnn')

        # [B, len(enc_filter_sizes), enc_num_filters]
        outputs = tf.concat(conv_outputs, axis=1)
        # [B, enc_num_filters]
        _states = tf.reduce_max(outputs, axis=1, keepdims=False)
        states = tf.contrib.rnn.LSTMStateTuple(c=_states, h=_states)
        return outputs, states
    elif model_configs.encoder_type == 'transformer':
        all_encoder_layers, sequence_output = encoder_modules.transformer_encoder(bert_config, is_training,
                                                                                  embedding_inputs, input_ids,
                                                                                  input_mask=input_mask,
                                                                                  token_type_ids=None,
                                                                                  scope=name + '_transformer')
        return all_encoder_layers, sequence_output
    else:
        raise NotImplementedError



