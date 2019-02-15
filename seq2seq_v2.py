# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
Linear = rnn_cell_impl._Linear  # pylint: disable=protected-access,invalid-name
def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])#### 做映射变化，从rnn的output size 映射到输入的inputsize
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol) ####从embedding矩阵中取出当前分类结果的词向量
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev) ####梯度下降词向量
    return emb_prev

  return loop_function


####rnn_decoder 输入是二维tensor，是已经embedding了的输入，embedding长度为input_size
def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initialstate  batch size x statesize,batch的每个init是不一样的？
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
    返回是list 长度为sequence length，每个item为batchsize x ouputsize
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):###遍历每个时间步
      if loop_function is not None and prev is not None:####prev is None表示第一个时间步，即输入为Go，即 当loopfunction不为空时，只有enumerate中的第一个inp会用到，后续的inp由loop function 产生
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i) ###当loop function 不为空
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)####cell.call(inp,state)
      outputs.append(output)###保存每个output结果
      if loop_function is not None:####loop function 不为空，则一般在测试阶段，
        prev = output
  return outputs, state


def basic_rnn_seq2seq(encoder_inputs,
                      decoder_inputs,
                      cell,
                      dtype=dtypes.float32,
                      scope=None):
  """Basic RNN sequence-to-sequence model.

  This model first runs an RNN to encode encoder_inputs into a state vector,
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell type, but don't share parameters.
  basic rnn  incoder decoder是一样的rnn结构，不同的网络

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size]. encoder的输入 时间步x[batchsize,inputsize]
    decoder_inputs: A list of 2D Tensors [batch_size x input_size]. decoder的输入 时间步x[batchsize,inputsize]
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size. cell的结构
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with  outputs  时间步x[batchsize,inputsize]
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in the final time-step. state 是最后一个时间步的state
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
    enc_cell = copy.deepcopy(cell) ####deeocopy一个网络
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)###encoder
    return rnn_decoder(decoder_inputs, enc_state, cell)####decoder，encoder的enc_state作为decoder的stateinit


def tied_rnn_seq2seq(encoder_inputs,
                     decoder_inputs,
                     cell,
                     loop_function=None,
                     dtype=dtypes.float32,
                     scope=None):
  """RNN sequence-to-sequence model with tied encoder and decoder parameters.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell and share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol), see rnn_decoder for details.
    dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
    scope = scope or "tied_rnn_seq2seq"
    _, enc_state = rnn.static_rnn(
        cell, encoder_inputs, dtype=dtype, scope=scope)
    variable_scope.get_variable_scope().reuse_variables()
    return rnn_decoder(
        decoder_inputs,
        enc_state,
        cell,
        loop_function=loop_function,
        scope=scope)


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option. embedding  pure decoding option

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs). 时间步x[batchsize，1]
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.字典长度
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has  projection 的输出，w=[outputsize,vocab_size]===>projection的分类结果=》argmax或beamsearch=》embedding look up 得到 vector
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.

    feed_previous: Boolean; if True, only the first of decoder_inputs will be ####greedy decoder 其实是找argmax，更好的有beam search， embedding lookup 从previous_ouput的最大编号中找到对应的embedding vector
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)), 通过embedding lookup 找到 输入的 vector
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).

    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"  ####只有第一个decoder inputs 会做update vector
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when  decoder 输出为 时间步x[batchsize,output size]
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]####在decoder部分不做ouput projection 输出维时间步x[batchsize,vocab size]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (embedding_ops.embedding_lookup(embedding, i)
               for i in decoder_inputs) ####遍历decoder input 通过embedding lookup 得到vector
    return rnn_decoder(
        emb_inp, initial_state, cell, loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
  """Embedding RNN sequence-to-sequence model.
  ####encoder decoder embedding 使用不同的embedding 矩阵

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs. decoder 的init state 是encoder最后一个时间步的state

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size]. encoder 输入，时间步X[batchsize,1]，每个输入是编号，编号转为embedding vector
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size]. decoder 输入，时间步X[batchsize,1]
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side. encoder 字典长度
    num_decoder_symbols: Integer; number of symbols on the decoder side. decoder 字典长度，这两个值一般一样，用同一个字典
    embedding_size: Integer, the length of the embedding vector for each symbol. embedding size，vector size大小

    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.####output projection，把cell的输出project到vocab 大小，再通过argmax 得到最大的编号，再用embedding look up 得到对应vector

    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first ####feed previous为True， 在test阶段生成一个序列，
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).

    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The 时间步X[batchsize,outputsize]
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is  None.此处描述和output projection 参数有出入? 似乎是is Not None做映射，符合逻辑
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size]. states是时间步Xstate，此处与basic seq2seq不同
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    # Encoder.
    encoder_cell = copy.deepcopy(cell) ####独立的encoder 和decoder
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype) ####做encoder，输入是经过EmbeddingWrapper后的encoder cell，输出 encoder state ，作为decoder的init state

    # Decoder.
    if output_projection is None:#####在output projection为None 时，做outputprojectionwrapper,此处的逻辑稍显混乱，意思是outputprojection is None，做一个默认的全vocab的映射？若不是None，做一个设定size的映射？
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):####feed previous 为True，一般为测试阶段，需要ouput projection和feed previous
      return embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
          ####此处做一个decoder， 返回的state是所有时间步的state
        outputs, state = embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),#####cond feed previous 为真，则参数为真 一般为测试，
                                              lambda: decoder(False))###若feed previous 为假，参数为假，一般为训练
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]####截取后面部分
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list) ####这些操作是对shape的形状处理
    return outputs_and_state[:outputs_len], state #####最终返回outputs和 state 都是时间步长度的list


def embedding_tied_rnn_seq2seq(encoder_inputs,
                               decoder_inputs,
                               cell,
                               num_symbols,
                               embedding_size,
                               num_decoder_symbols=None,
                               output_projection=None,
                               feed_previous=False,
                               dtype=None,
                               scope=None):
  """Embedding RNN sequence-to-sequence model with tied (shared) parameters.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_symbols x input_size]). Then it runs an RNN to encode embedded
  encoder_inputs into a state vector. Next, it embeds decoder_inputs using
  the same embedding. Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs. The decoder output is over symbols
  from 0 to num_decoder_symbols - 1 if num_decoder_symbols is none; otherwise it
  is over 0 to num_symbols - 1.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_symbols: Integer; number of symbols for both encoder and decoder.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_decoder_symbols: Integer; number of output symbols for decoder. If
      provided, the decoder output is over symbols 0 to num_decoder_symbols - 1.
      Otherwise, decoder output is over symbols 0 to num_symbols - 1. Note that
      this assumes that the vocabulary is set up such that the first
      num_decoder_symbols of num_symbols are part of decoding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the initial RNN states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_tied_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_symbols] containing the generated
        outputs where output_symbols = num_decoder_symbols if
        num_decoder_symbols is not None otherwise output_symbols = num_symbols.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(
      scope or "embedding_tied_rnn_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    if output_projection is not None:
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable(
        "embedding", [num_symbols, embedding_size], dtype=dtype)

    emb_encoder_inputs = [
        embedding_ops.embedding_lookup(embedding, x) for x in encoder_inputs
    ]
    emb_decoder_inputs = [
        embedding_ops.embedding_lookup(embedding, x) for x in decoder_inputs
    ]

    output_symbols = num_symbols
    if num_decoder_symbols is not None:
      output_symbols = num_decoder_symbols
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, output_symbols)

    if isinstance(feed_previous, bool):
      loop_function = _extract_argmax_and_embed(embedding, output_projection,
                                                True) if feed_previous else None
      return tied_rnn_seq2seq(
          emb_encoder_inputs,
          emb_decoder_inputs,
          cell,
          loop_function=loop_function,
          dtype=dtype)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      loop_function = _extract_argmax_and_embed(
          embedding, output_projection, False) if feed_previous_bool else None
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = tied_rnn_seq2seq(
            emb_encoder_inputs,
            emb_decoder_inputs,
            cell,
            loop_function=loop_function,
            dtype=dtype)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    # Calculate zero-state to know it's structure.
    static_batch_size = encoder_inputs[0].get_shape()[0]
    for inp in encoder_inputs[1:]:
      static_batch_size.merge_with(inp.get_shape()[0])
    batch_size = static_batch_size.value
    if batch_size is None:
      batch_size = array_ops.shape(encoder_inputs[0])[0]
    zero_state = cell.zero_state(batch_size, dtype)
    if nest.is_sequence(zero_state):
      state = nest.pack_sequence_as(
          structure=zero_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state

#####带attention的decoder attention is all you need!
def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.
    概括：
    attention look up attention states(encoder hidden state),focusing on a few entries

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.
  attention 对复杂seq2seq tasks

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states：是编码器的hidden state attn_length是时间步，attn_size，是h的向量长度，
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].


    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.####output size 是否再对output 做映射

    num_heads: Number of attention heads that read from attention_states. multi heads

    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of  ####时间步X[batchsize,outputsize]
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state)) ####new attn的计算
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size]. 返回final state

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value ###时间步
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value ####向量长度

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")) ###做卷积
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

    state = initial_state

    def attention(query): ###query为decoder的hidden state
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = Linear(query, attention_vec_size, True)(query) #####使用Linear 对query 做变换，变换成attention_vec_size
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])###reshape[batch_size,1,1,attention_vec_size]
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])###公式一
          a = nn_ops.softmax(s)###公式二
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum( ####公式三
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)###如果initial_state_attention存在，则调用attention函数对attns初始化
    for i, inp in enumerate(decoder_inputs):####对每个decoder inputs计算
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:#####主要在测试阶段
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      inputs = [inp] + attns ####将atten和input  concatenate
      x = Linear(inputs, input_size, True)(inputs)####通过Linear 函数转换，inputs由inp和attns catenatenate而来，故使用Linear做转换维度至input_size
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:####在第一步时候，设置variable scope
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        inputs = [cell_output] + attns ####concatenate ouput和attns，cell output与attns concatenate
        output = Linear(inputs, output_size, True)(inputs)###Linear 做维度变换到output size
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
        array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


def one2many_rnn_seq2seq(encoder_inputs,
                         decoder_inputs_dict,
                         enc_cell,
                         dec_cells_dict,
                         num_encoder_symbols,
                         num_decoder_symbols_dict,
                         embedding_size,
                         feed_previous=False,
                         dtype=None,
                         scope=None):
  """One-to-many RNN sequence-to-sequence model (multi-task).

  This is a multi-task sequence-to-sequence model with one encoder and multiple
  decoders. Reference to multi-task sequence-to-sequence learning can be found
  here: http://arxiv.org/abs/1511.06114

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs_dict: A dictionary mapping decoder name (string) to
      the corresponding decoder_inputs; each decoder_inputs is a list of 1D
      Tensors of shape [batch_size]; num_decoders is defined as
      len(decoder_inputs_dict).
    enc_cell: tf.nn.rnn_cell.RNNCell defining the encoder cell function and
      size.
    dec_cells_dict: A dictionary mapping encoder name (string) to an
      instance of tf.nn.rnn_cell.RNNCell.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols_dict: A dictionary mapping decoder name (string) to an
      integer specifying number of symbols for the corresponding decoder;
      len(num_decoder_symbols_dict) must be equal to num_decoders.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "one2many_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs_dict, state_dict), where:
      outputs_dict: A mapping from decoder name (string) to a list of the same
        length as decoder_inputs_dict[name]; each element in the list is a 2D
        Tensors with shape [batch_size x num_decoder_symbol_list[name]]
        containing the generated outputs.
      state_dict: A mapping from decoder name (string) to the final state of the
        corresponding decoder RNN; it is a 2D Tensor of shape
        [batch_size x cell.state_size].

  Raises:
    TypeError: if enc_cell or any of the dec_cells are not instances of RNNCell.
    ValueError: if len(dec_cells) != len(decoder_inputs_dict).
  """
  outputs_dict = {}
  state_dict = {}

  if not isinstance(enc_cell, rnn_cell_impl.RNNCell):
    raise TypeError("enc_cell is not an RNNCell: %s" % type(enc_cell))
  if set(dec_cells_dict) != set(decoder_inputs_dict):
    raise ValueError("keys of dec_cells_dict != keys of decodre_inputs_dict")
  for dec_cell in dec_cells_dict.values():
    if not isinstance(dec_cell, rnn_cell_impl.RNNCell):
      raise TypeError("dec_cell is not an RNNCell: %s" % type(dec_cell))

  with variable_scope.variable_scope(
      scope or "one2many_rnn_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    # Encoder.
    enc_cell = core_rnn_cell.EmbeddingWrapper(
        enc_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    for name, decoder_inputs in decoder_inputs_dict.items():
      num_decoder_symbols = num_decoder_symbols_dict[name]
      dec_cell = dec_cells_dict[name]

      with variable_scope.variable_scope("one2many_decoder_" + str(
          name)) as scope:
        dec_cell = core_rnn_cell.OutputProjectionWrapper(
            dec_cell, num_decoder_symbols)
        if isinstance(feed_previous, bool):
          outputs, state = embedding_rnn_decoder(
              decoder_inputs,
              encoder_state,
              dec_cell,
              num_decoder_symbols,
              embedding_size,
              feed_previous=feed_previous)
        else:
          # If feed_previous is a Tensor, we construct 2 graphs and use cond.
          def filled_embedding_rnn_decoder(feed_previous):
            """The current decoder with a fixed feed_previous parameter."""
            # pylint: disable=cell-var-from-loop
            reuse = None if feed_previous else True
            vs = variable_scope.get_variable_scope()
            with variable_scope.variable_scope(vs, reuse=reuse):
              outputs, state = embedding_rnn_decoder(
                  decoder_inputs,
                  encoder_state,
                  dec_cell,
                  num_decoder_symbols,
                  embedding_size,
                  feed_previous=feed_previous)
            # pylint: enable=cell-var-from-loop
            state_list = [state]
            if nest.is_sequence(state):
              state_list = nest.flatten(state)
            return outputs + state_list

          outputs_and_state = control_flow_ops.cond(
              feed_previous, lambda: filled_embedding_rnn_decoder(True),
              lambda: filled_embedding_rnn_decoder(False))
          # Outputs length is the same as for decoder inputs.
          outputs_len = len(decoder_inputs)
          outputs = outputs_and_state[:outputs_len]
          state_list = outputs_and_state[outputs_len:]
          state = state_list[0]
          if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(
                structure=encoder_state, flat_sequence=state_list)
      outputs_dict[name] = outputs
      state_dict[name] = state

  return outputs_dict, state_dict


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(labels=target, logits=logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last "
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last "
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses
