# seq2seq
char_rnn_trump_tweet.py  堆叠两个gru 生成trump tweet

seq2seq_v1.py 实现encoder decoder结构，预测时间序列，每个encoder和deocer为两个lstm堆叠，部分参考tensorflow seq2seq代码
seq2seq_v2.py是tensorflow的源码注释，主要注释的方法有：
rnn_decoder()
basic_rnn_seq2seq()
embedding_rnn_decoder()
embedding_rnn_seq2seq()
attention_decoder()
embedding_attention_seq2seq()


