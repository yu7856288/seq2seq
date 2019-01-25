# -*-encoding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import sys
sys.path.append('..')
import time
import tensorflow as tf
import utils

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab] ####对vocab对单词做index，从1开始计数,0是留空，用于补充0


def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array]) ####把index转换为单词，index需要减1


def read_data(filename, vocab, sequence_length, overlap):
    lines = [line.strip() for line in open(filename, 'r', encoding='utf-8').readlines()]
    while True:
        random.shuffle(lines)
        for text in lines:
            text = vocab_encode(text, vocab)####转为序号
            for start in range(0, len(text) - sequence_length, overlap):
                chunk = text[start: start + sequence_length]
                chunk += [0] * (sequence_length - len(chunk))###不满sequence_length size的chunk 则补0
                yield chunk


def read_batch(stream, batch_size):
    batch = []
    for element in stream: ####for in 有next功能
        batch.append(element)
        if len(batch) == batch_size:
            yield batch ###yield返回结果
            batch = [] ###从此处执行新的语句
    yield batch


class CharRNN(object):
    def __init__(self, model):
        self.model = model
        self.path = 'data/' + model + '.txt'
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 256]
        self.batch_size = 64
        self.lr = 0.0003
        self.skip_step = 100  ###评估间隔
        self.sequence_length = 50  # 时间步 长度
        self.len_generated = 200  ####生成的句子的长度
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step') ####计数器

        #####获得字典vocab 共87个
        if 'trump' in model:
            self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
        else:
            self.vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          "\\^_abcdefghijklmnopqrstuvwxyz{|}")

        self.seq = tf.placeholder(tf.int32, [None, None]) ###输入batch  0维可能不足batch size x seqence_length


    def create_rnn(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes] ####堆叠 GRUcell 用两层
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]
        zero_states = cells.zero_state(batch, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])
                               for state in zero_states])###  第一维为None ，可使用不同的batch 大小 将list 转tuple,state需要tuple

        # this line to calculate the real length of seq
        # all seq are padded to be of the same length, which is sequence_length
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)####z这行代码可能有误
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state) ###output与GRU的隐含值的h一致
        ###output shape [64,50,256]

    def create_model(self):
        seq = tf.one_hot(self.seq, len(self.vocab)) ###把字典转为one hot 编码
        #seq shape [64,50,87]
        self.create_rnn(seq)
        ###gru输出层和隐含h一致，需要做transform，用tf.layers.dense做维度变换
        self.logits = tf.layers.dense(self.output, len(self.vocab), activation=None)###output 做一个dense的transform，输出的units个数为字典长度，下一步做softMax
        ####logits shape [64,50,87]
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], #### 预测序列，y0,y1,...y49,因为最后一个从x49取得，则不需要这个y49
                                                       labels=seq[:, 1:]) ####labels是输出的序列，原序列是x0，x1，x2,...,则预测序列维x1,x2,x3,...
        self.loss = tf.reduce_sum(loss)
        # sample the next character from Maxwell-Boltzmann Distribution
        # with temperature temp. It works equally well without tf.exp
        self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0] ###output Maxwell-Boltzmann Distribution 采样得到一个sample
        ###sample shape [64,1] 或者就是[64]

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs/gist', sess.graph)
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.model + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            stream = read_data(self.path, self.vocab, self.sequence_length, overlap=self.sequence_length // 2)
            data = read_batch(stream, self.batch_size)
            while True:
                batch = next(data)
                ###batch shape [64,50]
                output_,logit_=sess.run([self.output,self.logits],feed_dict={self.seq:batch})
                # for batch in read_batch(read_data(DATA_PATH, vocab)):
                batch_loss, _ = sess.run([self.loss, self.opt], {self.seq: batch})
                if (iteration + 1) % self.skip_step == 0:
                    print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                    self.online_infer(sess)
                    start = time.time()
                    checkpoint_name = 'checkpoints/' + self.model + '/char-rnn'
                    if min_loss is None:
                        saver.save(sess, checkpoint_name, iteration)
                    elif batch_loss < min_loss:
                        saver.save(sess, checkpoint_name, iteration)
                        min_loss = batch_loss
                iteration += 1

    def online_infer(self, sess):
        """ Generate sequence one character at a time, based on the previous character
        """
        for seed in ['Hillary', 'I', 'R', 'T', '@', 'N', 'M', '.', 'G', 'A', 'W']:
            sentence = seed
            state = None
            for _ in range(self.len_generated):
                batch = [vocab_encode(sentence[-1], self.vocab)] ###取得seed的最后一个字母，编码 外套一层，则batch为两维
                 ##此处batch [1,1]
                feed = {self.seq: batch}
                if state is not None:  # for the first decoder step, the state is None
                    for i in range(len(state)):
                        feed.update({self.in_state[i]: state[i]})
                index, state,logits_= sess.run([self.sample, self.out_state,self.logits], feed) ####执行一次，得到一个sample，需要更新in_state,因为这个是序列生成，只能一步一步跑，没法用dynamic_rnn
                print(logits_.shape)
                sentence += vocab_decode(index, self.vocab)
            print('\t' + sentence)


def main():
    model = 'trump_tweets'
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/' + model)
    lm = CharRNN(model)
    lm.create_model()
    lm.train()


if __name__ == '__main__':
    main()
