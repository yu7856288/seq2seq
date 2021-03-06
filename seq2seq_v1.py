import copy
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x = np.linspace(0, 30, 105)
y = 2 * np.sin(x)

l1, = plt.plot(x[:85], y[:85], 'y', label = 'training samples')
l2, = plt.plot(x[85:], y[85:105], 'c--', label = 'test samples')
plt.legend(handles = [l1, l2], loc = 'upper left')
plt.show()

###模拟噪声
train_y = y.copy()
noise_factor = 0.5
train_y += np.random.randn(105) * noise_factor

l1, = plt.plot(x[:85], train_y[:85], 'yo', label = 'training samples')
plt.plot(x[:85], y[:85], 'y:')
l2, = plt.plot(x[85:], train_y[85:], 'co', label = 'test samples')
plt.plot(x[85:], y[85:], 'c:')
plt.legend(handles = [l1, l2], loc = 'upper left')
plt.show()

encoder_seq_len = 15
decoder_seq_len = 20

x = np.linspace(0, 30, 105)
train_data_x = x[:85]

def true_signal(x):
    y = 2 * np.sin(x)
    return y

def noise_func(x, noise_factor = 1):
    return np.random.randn(len(x)) * noise_factor

def generate_y_values(x):
    return true_signal(x) + noise_func(x)

def generate_train_samples(x = train_data_x, batch_size = 10, encoder_seq_len = encoder_seq_len, decoder_seq_len = decoder_seq_len):

    total_start_points = len(x) - encoder_seq_len - decoder_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size) ####一个序列，batch size大小，

    input_seq_x = [x[i:(i+encoder_seq_len)] for i in start_x_idx] ###encoder_seq_len 长度的 数据
    output_seq_x = [x[(i+encoder_seq_len):(i+encoder_seq_len+decoder_seq_len)] for i in start_x_idx] ####ouput_seq_len长度的输出数据（decoder）

    input_seq_y = [generate_y_values(x) for x in input_seq_x] ###为encoder的input 为一个list,使用static_rnn， 输入要求
    output_seq_y = [generate_y_values(x) for x in output_seq_x]###为decoder的input为一个list，使用static_rnn

    return np.array(input_seq_y), np.array(output_seq_y) ###返回np array

encoder_seq, decoder_seq = generate_train_samples(batch_size=10) ####模拟生成encoder和decoder的数据

###噪声数据可视化
results = []
for i in range(100):
    temp = generate_y_values(x)
    results.append(temp)
results = np.array(results)
print(results.shape)

for i in range(100):
    l1, = plt.plot(results[i].reshape(105, -1), 'co', lw = 0.1, alpha = 0.05, label = 'noisy training data')####把批量噪声数据可视化，此处画100条数据

l2, = plt.plot(true_signal(x), 'm', label = 'hidden true signal')
plt.legend(handles = [l1, l2], loc = 'lower left')
plt.show()
#####################################################################################################
#####################以上为数据测试#####################################################################
#####################################################################################################


####基本的rnn模型参数
## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003

## Network Parameters
# length of input signals
encoder_seq_len = 15
# length of output signals
decoder_seq_len = 20
# size of LSTM Cell
hidden_dim = 64
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5 ###梯度clip 阈值

epoch = 100 ####
batch_size = 16
KEEP_RATE = 0.5
train_losses = []
val_losses = []




def build_graph(feed_previous = False):

    tf.reset_default_graph()

    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    ###weights 和biases用于decoder target的transform
    weights = {
        'out': tf.get_variable('Weights_out', shape = [hidden_dim, output_dim],dtype = tf.float32, initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', shape = [output_dim], dtype = tf.float32, initializer = tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):####最外层的variable scope seq2seq
        # Encoder: inputs 生成一个encoder_inputs的list,用于static rnn
        encoder_inputs = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(encoder_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [ ###target_seq 是目标输出，是要预测的值
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(decoder_seq_len)
        ]
        decoder_inputs = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1] ####decider input 大小也是20,

        with tf.variable_scope('LSTMCell'):####第二层variable scope lstm cell 的rnn结构
            layers = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    layers.append(tf.nn.rnn_cell.LSTMCell(hidden_dim))
            cells = tf.nn.rnn_cell.MultiRNNCell(layers)

        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cells,
                        loop_function=None,
                        scope=None):
          with tf.variable_scope(scope or "rnn_decoder"):####在第三层一个decoder的variable scope
            state = initial_state ####initial_state 为encoder_state的语义向量，用在decoder的第一个输入中，作为他的 初始state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
                ####遍历所有的decoder_inputs，
                ####decoder_input 为go,y1,y2,y3,...,y19
                ####target output 为y1,y2,y3,y4...,y20
                ####hidden state 为 enc_state,dec_h1,dec_h2,dec_h3,...,dec_h19
                ###decoder_output y1_hat,y2_hat,y3_hat,...,y20_hat
                ###loss 比较 yi_hat与 yi的square mean
              if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                tf.get_variable_scope().reuse_variables()
              output, state = cells.call(inp, state) #
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=tf.float32,
                              scope=None):

          with tf.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell) ####deep copy 两个不同的cell
            _, enc_state = tf.nn.static_rnn(enc_cell, encoder_inputs, dtype=dtype) ####encoder
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function) ####decoder 测试阶段使用
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell) ####此处的output 没有做过tranform，仍然和hidden output 一致，训练阶段使用

        def _loop_function(prev, _):####output 到实际的targe的output 做一个tranform，rnn cell的output其实和hidden ouput一致

          return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            encoder_inputs,
            decoder_inputs,
            cells,
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs] ####decoder output 转换

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):#####求target ouput和 decoder  实际的output的 mean square
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables(): #####遍历所有trainable variables
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var)) ####l2 loss

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):####optimizer
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        encoder_inputs = encoder_inputs,
        target_seq = target_seq,
        train_op = optimizer,
        loss=loss,
        saver = saver,
        reshaped_outputs = reshaped_outputs,
        )



x = np.linspace(0, 30, 105)
train_data_x = x[:85]

rnn_model = build_graph(feed_previous=False) ####此时 feed_previous 为false

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size) ####取得batch size的  input ouptut
        feed_dict = {rnn_model['encoder_inputs'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(encoder_seq_len)} ####输入encoder
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(decoder_seq_len)}) ####输入decoder，作为targe
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'univariate_ts_model0'))

print("Checkpoint saved at: ", save_path)



test_seq_input = true_signal(train_data_x[-15:])

rnn_model = build_graph(feed_previous=True) ####此时feed_previous 为true

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    saver = rnn_model['saver']().restore(sess, os.path.join('./', 'univariate_ts_model0'))

    feed_dict = {rnn_model['encoder_inputs'][t]: test_seq_input[t].reshape(1,1) for t in range(encoder_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(decoder_seq_len)}) ####在测试阶段，target_seq设置为0，方便ops统一，实际并不为0
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    final_preds = np.concatenate(final_preds, axis = 1)


l1, = plt.plot(range(85), true_signal(train_data_x[:85]), label = 'Training truth')
l2, = plt.plot(range(85, 105), y[85:], 'yo', label = 'Test truth')
l3, = plt.plot(range(85, 105), final_preds.reshape(-1), 'ro', label = 'Test predictions')
plt.legend(handles = [l1, l2, l3], loc = 'lower left')
plt.show()
