#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#�������������������������������������������ݡ�������������������������������������������
f = open('/Users/liyangyang/Downloads/fund_data.csv')
df = pd.read_csv(f)
data = np.array(df['price']) #��ȡ��߼�����
data = data[::-1] #��ת�������ݰ��������Ⱥ�˳������

#����ͼչʾdata
plt.figure()
plt.plot(data)
plt.show()
normalize_data = (data - np.mean(data))/np.std(data)#��׼��
normalize_data = normalize_data[:,np.newaxis] #����ά��

#�γ�ѵ����
#���ó���
time_step = 20
rnn_unit = 10 #���ز㵥Ԫ
batch_size = 10
input_size = 1
output_size = 1
lr = 0.0006
train_x, train_y = [], []
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i: i+time_step]
    y = normalize_data[i+1: i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())



#�������������������������������������������������������������������������������������
X = tf.placeholder(tf.float32, [None, time_step, input_size])
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
weights = {
    'in':tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,1]))
}
biases = {
    'in':tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out':tf.Variable(tf.constant(0.1,shape=[1,]))
}


#����lstm������
def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size]) #��Ҫ��tensorת��2ά���м��㣬�����Ľ����Ϊ���ز������
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit]) #��tensorת��3ά����Ϊlstm cell������
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse = tf.AUTO_REUSE)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit]) #��Ϊ����������
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

#ѵ��ģ��
def train_lstm(train_begin, train_end):
    global batch_size
    pred,_ = lstm(batch_size)
    #��ʧ����
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #�ظ�ѵ��10000��
        for i in range(10):
            step = 0
            start = 0
            end = start + batch_size
            while(end < len(train_x)):
                _,loss_ = sess.run([train_op, loss], feed_dict={X:train_x[start:end], Y:train_y[start:end]})
                start += batch_size
                end = start + batch_size
                #ÿ10������һ�β���
                if step%10 == 0:
                    print (i, step, loss_)
                    print ("����ģ�ͣ�", saver.save(sess, '/Users/liyangyang/PycharmProjects/gupiao/stock.model'))
                step += 1

#Ԥ��ģ��
def prediction(test_begin):
    pred, _ = lstm(1) #Ԥ��ʱֻ���롾1�� time_step. input_size]�Ĳ�������
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #�����ָ�
        module_file = tf.train.latest_checkpoint('/Users/liyangyang/PycharmProjects/gupiao/')
        saver.restore(sess, module_file)
        #ȡѵ�������һ��Ϊ�������ԡ�shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        predict = []
        #�õ�֮��Ԥ���5�����
        for i in range(50):
            next_seq = sess.run(pred, feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            # ÿ�εõ����һ��ʱ�䲽��Ԥ��������֮ǰ�����ݼ���һ���γ��µĲ�������
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        #������ͼ��ʾ���
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color = 'b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color = 'r')
        plt.show()

train_begin = 0
train_end = len(normalize_data)*0.8

train_lstm(train_begin=train_begin, train_end=train_end)
prediction(test_begin=train_end)
