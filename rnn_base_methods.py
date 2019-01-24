
import tensorflow as tf
import numpy as np 
batch_size=2 #批处理大小
 
hidden_size=3 #隐藏层神经元
 
max_time=5 #最大时间步长
 
depth=6 #输入层神经元数量，如词向量维度

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


print("Multi GRU")
layers=[tf.nn.rnn_cell.GRUCell(num_units=units) for units in range(3,6)]
cells=tf.nn.rnn_cell.MultiRNNCell(layers)
h0=cells.zero_state(batch_size=batch_size,dtype=tf.float32)
inputs=tf.Variable(tf.random.truncated_normal(shape=[batch_size,depth]))
output,h1=cells.call(inputs,state=h0)
dynamic_inputs=tf.Variable(tf.random.truncated_normal(shape=[batch_size,max_time,depth]))
dynamic_outpus,dynamic_h=tf.nn.dynamic_rnn(cells,inputs=dynamic_inputs,initial_state=h0)
print(output.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(output)
    print(h1)
    dynamic_inputs_,dynamic_h_=sess.run([dynamic_inputs,dynamic_h])
    print(dynamic_inputs_.shape)
    print(dynamic_h_)





print("multi lstm  ")
inputs=tf.Variable(tf.random.truncated_normal(shape=[batch_size,depth]))
dynamic_inputs=tf.Variable(tf.random.truncated_normal(shape=[batch_size,max_time,depth]))
layers=[tf.nn.rnn_cell.BasicLSTMCell(num_units=units) for units in range(5,8)]
cells=tf.nn.rnn_cell.MultiRNNCell(layers)
init_state=cells.zero_state(batch_size=batch_size,dtype=tf.float32)

output,h1=cells.call(inputs,state=init_state)
bi_lstm_output,bi_lstm_h=tf.nn.dynamic_rnn(cells,inputs=dynamic_inputs,initial_state=init_state)
print(bi_lstm_output.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(output.shape)
    print(bi_lstm_output.shape)
    print(output.shape)



print("bi lstm")
inputs=tf.Variable(tf.random.truncated_normal(shape=[batch_size,max_time,depth]))
lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
output,output_state=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_,state_=sess.run([output,output_state])
    output_fw, output_bw = output_
    states_fw, states_bw = state_
    print(output_fw.shape)
    print(output_bw.shape)
    print(states_fw)
    print(states_bw)
