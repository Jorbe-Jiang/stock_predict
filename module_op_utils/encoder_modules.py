import tensorflow as tf
import numpy as np
from module_op_utils import bert_module

tf.set_random_seed(2019)
np.random.seed(2019)


def cnn_encoder_v1(embedding_inputs, num_filters, kernel_size, ret_pooled=True, name='conv_enc_layer'):
    """
    cnn encoder: one CNN layer and w/wo max pooling
    :param embedding_inputs: [B * T * D]
    :param num_filters: scalar
    :param kernel_size: scalar
    :param ret_pooled: boolean
    :param name: string
    :return: gmp/conv
    """
    # CNN layer [B, T_hat, num_filters], 如果padding=SAME, 则T_hat=T， T为原始序列长度
    conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, strides=1, padding='valid', name=name)
    gmp = tf.reduce_max(conv, axis=1, keepdims=False, name=name + 'gmp')
    if ret_pooled:
        return gmp
    else:
        return conv


def cnn_encoder(embedding_inputs, emb_size, num_filters, filter_sizes=[3, 5], max_pooling=False, name='conv_enc_layer'):
    """
    cnn encoder: one CNN layer and w/wo max pooling
    :param embedding_inputs: tensor
    :param emb_size: scalar
    :param num_filters: scalar
    :param filter_sizes: list, [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    :param max_pooling: boolean
    :param name: string
    :return: tensor
    """
    # filter_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    with tf.variable_scope(name):
        # Create a convolution layer for each filter size
        conv_outputs = []
        for filter_size in filter_sizes:
            with tf.variable_scope('conv-%s' % filter_size):
                filter_shape = [
                    filter_size, emb_size, num_filters
                ]
                W = tf.get_variable(
                    name='W', initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=[num_filters]))
                # [B, T, num_filters], 因为padding是SAME而且stride=1，所以第一维还是序列长度T
                conv = tf.nn.conv1d(embedding_inputs, W, stride=1, padding='SAME', name='conv')

                if max_pooling:
                    # [B, 1, num_filters]
                    conv = tf.reduce_max(conv, axis=1, keepdims=True)

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                conv_outputs.append(h)

        # num_filters_total = num_filters * len(filter_sizes)
        #
        # # [B, T, num_filters * len(filter_sizes)], if max_pooling=True: T=1
        # h_conv = tf.concat(conv_outputs, axis=2)
        # # [B*T, num_filters_total], if max_pooling=True: T=1
        # h_conv_flat = tf.reshape(h_conv, [-1, num_filters_total])
        # return h_conv, h_conv_flat
        return conv_outputs


def rnn_encoder(embedding_inputs, sequence_length, hidden_size, num_layers, batch_size, is_training=True, keep_prob=1.0,
                rnn_type='lstm', birnn=False, name='rnn_enc_layer', reuse=None):
    if not is_training:
        keep_prob = 1.0

    if rnn_type != 'lstm':
        assert not birnn

    with tf.variable_scope(name, reuse=reuse):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(hidden_size)

        if rnn_type == 'lstm':
            enc_cell = lstm_cell
        else:
            enc_cell = gru_cell

        if is_training and keep_prob < 1.0:
            def enc_cell():
                if rnn_type == 'lstm':
                    if birnn:
                        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)
                    else:
                        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)
                else:
                    return tf.contrib.rnn.DropoutWrapper(gru_cell(), output_keep_prob=keep_prob)

        if birnn:
            if num_layers == 1:
                rnn_cell = enc_cell()
                # outputs: tuple([B, T, hidden_size], [B, T, hidden_size])
                outputs, states_tuple = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell, cell_bw=rnn_cell, dtype=tf.float32,
                                                                        inputs=embedding_inputs, sequence_length=sequence_length)
                outputs = tf.concat(outputs, 2)  # [B, T, 2*hidden_size]
                # states: LSTMStateTuple([B, hidden_size*2], [B, hidden_size*2])
                states = tf.contrib.rnn.LSTMStateTuple(
                    c=tf.concat([states_tuple[0].c, states_tuple[1].c], 1),
                    h=tf.concat([states_tuple[0].h, states_tuple[1].h], 1))
            else:
                stack_cells = [enc_cell() for _ in range(num_layers)]

                # outputs: [batch_size, T, hidden_size*2]
                outputs, fw_states_li, bw_states_li = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=stack_cells, cells_bw=stack_cells, inputs=embedding_inputs,
                    sequence_length=sequence_length, dtype=tf.float32)
                # states: num_layers-LSTMStateTuple([B, hidden_size*2], [B, hidden_size*2])
                states = tuple([tf.contrib.rnn.LSTMStateTuple(
                    c=tf.concat([fw_states_li[t].c, bw_states_li[t].c], 1),
                    h=tf.concat([fw_states_li[t].h, bw_states_li[t].h], 1)) for t in range(num_layers)])

                # print(states[0].c)
                # print(states[1].c)
                # print(states)
                # exit()

        else:
            # 多层rnn网络
            cells = [enc_cell() for _ in range(num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            initial_state = rnn_cell.zero_state(batch_size, tf.float32)
            # states: num_layers-LSTMStateTuple(c, h)
            outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs,
                                                sequence_length=sequence_length, initial_state=initial_state, dtype=tf.float32)

        return outputs, states


def transformer_encoder(bert_config, is_training, embedding_inputs, input_ids, input_mask=None, token_type_ids=None,
                        scope='transformer_encoder'):
    """
    调用bert模块的transformer结构来对文本序列进行编码，不需要token_type_ids，因为这里仅仅用到了bert里的transformer结构
    :param bert_config:
    :param is_training:
    :param embedding_inputs: [B, T, D]
    :param input_ids: [B, T]
    :param input_mask: [B, T]
    :param token_type_ids: None
    :param scope:
    :return: bert每层的序列编码输出以及最后一层的序列编码输出
    """
    bert_model = bert_module.BertModel(bert_config, is_training, embedding_inputs, input_ids,
                                       input_mask=input_mask, token_type_ids=token_type_ids, scope=scope)
    all_encoder_layers = bert_model.get_all_encoder_layers()
    # final hidden layer of BERT encoder: [batch_size, seq_len, hidden_size]
    sequence_output = bert_model.get_sequence_output()
    return all_encoder_layers, sequence_output
