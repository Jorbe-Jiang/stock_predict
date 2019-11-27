from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from module_op_utils import attention_utils
from module_op_utils import variational_dropout


def decoder_rnn(dec_input_ids, enc_outputs, enc_states, model_configs, is_training, enc_embedding=None,
                attention_option='luong', name='decoder_rnn', is_generator=False, use_gen_mode=False, temperature=1.0, reuse=None):
    """
    rnn解码器
    :param dec_input_ids: decoder输入ids
    :param enc_outputs: encoder序列输出
    :param enc_states: encoder最后一个时刻cell state
    :param model_configs:
    :param is_training:
    :param enc_embedding: encoder's word embedding
    :param attention_option: 注意力机制
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        if model_configs.encoder_type == 'birnn':
            dec_hidden_size = model_configs.enc_hidden_size * 2
        else:
            dec_hidden_size = model_configs.enc_hidden_size

        def dec_lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(dec_hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        dec_attn_cell = dec_lstm_cell

        if is_training and model_configs.dec_keep_prob < 1:
            def dec_attn_cell():
                return variational_dropout.VariationalDropoutWrapper(dec_lstm_cell(), model_configs.batch_size,
                                                                     dec_hidden_size, model_configs.dec_keep_prob,
                                                                     model_configs.dec_keep_prob)

        cell_dec = tf.contrib.rnn.MultiRNNCell([dec_attn_cell() for _ in range(model_configs.enc_num_layers)],
                                               state_is_tuple=True)

        # Hidden encoder states.
        hidden_vector_encodings = enc_outputs
        state_dec = enc_states

        if attention_option is not None:
            with tf.variable_scope('dec_attention', reuse=reuse):
                (dec_attention_keys, dec_attention_values, dec_attention_score_fn,
                 dec_attention_construct_fn) = attention_utils.prepare_attention(hidden_vector_encodings,
                                                                                 attention_option,
                                                                                 num_units=dec_hidden_size, reuse=reuse)

        def make_mask(keep_prob, units):
            random_tensor = keep_prob
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            random_tensor += tf.random_uniform(tf.stack([model_configs.batch_size, units]))
            return tf.floor(random_tensor) / keep_prob

        if is_training:
            dec_output_mask = make_mask(model_configs.dec_keep_prob, dec_hidden_size)

        token_sequence, logits_sequence, probs_sequence = [], [], []

        if not model_configs.seq2seq_share_embedding:
            dec_embedding = tf.get_variable('dec_embedding', [model_configs.vocab_size, dec_hidden_size])
        else:
            assert enc_embedding
            if model_configs.encoder_type == 'birnn':
                assert model_configs.emb_size == dec_hidden_size
                dec_embedding = enc_embedding

        dec_softmax_w = tf.matrix_transpose(dec_embedding)
        dec_softmax_b = tf.get_variable('dec_softmax_b', [model_configs.vocab_size])

        dec_rnn_inputs = tf.nn.embedding_lookup(dec_embedding, dec_input_ids)

        dec_rnn_outs = []

        fake = None
        for t in range(model_configs.max_seq_length):
            if t > 0:
                tf.get_variable_scope().reuse_variables()

            # Input to the Decoder.
            if t == 0:
                # Always provide the real input at t = 0.
                dec_rnn_inp = dec_rnn_inputs[:, t]

            else:
                dec_real_rnn_inp = dec_rnn_inputs[:, t]

                if is_training and not is_generator:
                    dec_rnn_inp = dec_real_rnn_inp
                    if model_configs.add_dec_emb_noise:
                        noise = tf.random_normal([model_configs.batch_size, model_configs.emb_size], seed=1111)
                        dec_rnn_inp += noise
                else:
                    fake_dec_rnn_inp = tf.nn.embedding_lookup(dec_embedding, fake)
                    dec_rnn_inp = fake_dec_rnn_inp

            # RNN.
            with tf.variable_scope('dec_rnn'):
                dec_rnn_out, state_dec = cell_dec(dec_rnn_inp, state_dec)

            if attention_option is not None:
                dec_rnn_out = dec_attention_construct_fn(dec_rnn_out, dec_attention_keys, dec_attention_values)

            if is_training:
                dec_rnn_out *= dec_output_mask

            dec_rnn_outs.append(dec_rnn_out)
            dec_logit = tf.nn.bias_add(tf.matmul(dec_rnn_out, dec_softmax_w), dec_softmax_b)
            dec_logit = dec_logit / tf.constant(temperature, dtype=tf.float32)
            probs = tf.nn.softmax(dec_logit, axis=1)

            # Output for Decoder.
            categorical = tf.contrib.distributions.Categorical(probs=probs)
            if use_gen_mode:
                fake = categorical.mode()
            else:
                fake = categorical.sample()
            # dec_log_prob = categorical.log_prob(fake)
            dec_output = fake

            # Add to lists.
            token_sequence.append(dec_output)
            probs_sequence.append(probs)
            logits_sequence.append(dec_logit)

    return tf.stack(token_sequence, axis=1), tf.stack(logits_sequence, axis=1), tf.stack(probs_sequence, axis=1)
