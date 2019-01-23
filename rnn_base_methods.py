

import tensorflow as tf
import numpy as np 
batch_size=2 #批处理大小
 
hidden_size=3 #隐藏层神经元
 
max_time=5 #最大时间步长
 
depth=6 #输入层神经元数量，如词向量维度
# basic rnn cell 和 lstm cell 的output不同， basic rnn output 和hidden state 相同，输出需transform,lstm
# 输出需softmax

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
basic_rnn_input_one_step=tf.Variable(tf.random_normal([batch_size,depth])) 
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

basic_rnn_input_steps=tf.Variable(tf.random_normal([batch_size,max_time,depth]))
 
# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
# defining initial state
# 'state' is a tensor of shape [batch_size, cell_state_size]
 

dynamic_basic_rnn_outputs, dynamic_baisc_rnn_states = tf.nn.dynamic_rnn(rnn_cell, basic_rnn_input_steps,initial_state=initial_state, dtype= tf.float32)
baisc_rnn_output,basic_rnn_state=rnn_cell.call(basic_rnn_input_one_step,state=initial_state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(dynamic_basic_rnn_outputs))
    print(sess.run(dynamic_baisc_rnn_states))





lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

dynamic_lstm_inputs = tf.placeholder(np.float32, shape=(batch_size, max_time,depth)) # 32 是 batch_size
lstm_input_one_step_input=tf.Variable(tf.random_normal([batch_size,depth])) 

dynamic_lstm_h0 = lstm_cell.zero_state(batch_size, np.float32) # 通过zero_state得到一个全0的初始状态

outputs,state=tf.nn.dynamic_rnn(lstm_cell,inputs=dynamic_lstm_inputs,initial_state=initial_state)
lstm_one_step_output,lstm_one_step_state=lstm_cell.call(lstm_input_one_step_input,dynamic_lstm_h0)
print(lstm_one_step_output)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs,feed_dict={dynamic_lstm_inputs:np.random.rand(batch_size,max_time,depth)}))



import tensorflow as tf
import numpy as np

batch_size=2 #批处理大小
 
hidden_size=3 #隐藏层神经元
 
max_time=5 #最大时间步长
depth=6
tf.set_random_seed=1234

# 每调用一次这个函数就返回一个BasicRNNCell

def get_a_cell(units):
   return tf.nn.rnn_cell.BasicRNNCell(num_units=units)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN

cells = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(units) for  units in range(3,6)]) # 3层RNN


print(cells.state_size) #  

# 使用对应的call函数

inputs = tf.placeholder(np.float32, shape=(batch_size, depth)) #  
dynamic_inputs=tf.placeholder(np.float32, shape=(batch_size,max_time, depth)) #  

h0 = cells.zero_state(batch_size, np.float32) # 通过zero_state得到一个全0的初始状态

output, h1 = cells.call(inputs, h0)

print(h1) # tuple中含有3个32x128的向量

dynamic_multi_rnn_outputs, dynamic_multi_rnn_states = tf.nn.dynamic_rnn(cells, dynamic_inputs,initial_state=h0, dtype= tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output,feed_dict={inputs:np.random.rand(batch_size,depth)}))
    print(sess.run(dynamic_inputs,feed_dict={dynamic_inputs:np.random.rand(batch_size,max_time,depth)}))
