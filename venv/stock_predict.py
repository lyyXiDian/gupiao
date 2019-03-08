#coding=gbk
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#���峣��
rnn_unit=10       #hidden layer units
input_size=2
output_size=2
lr=0.0006         #ѧϰ��

f=open('/Users/liyangyang/Downloads/apple.csv')
df=pd.read_csv(f)     #�����Ʊ����
data_0=df.iloc[:2521,1:3].values   #ȡ��2-4��
data = data_0[-1::-1, :] #����
# print (data)

train_begin = 0
train_end = 2500
data_train = data[train_begin:train_end]
data_test = data[2499:]
test_y = []
for d in data_test:
    test_y.append(d[0])
print('test y:', test_y)
mean = np.mean(data_train, axis=0)
std = np.std(data_train, axis=0)

def get_train_data(batch_size, time_step):
    # train_begin = 0
    batch_index = []
    # data_train = data[train_begin:]
    # mean = np.mean(data_train, axis=0)
    # std = np.std(data_train, axis=0)
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # ��׼��
    train_x, train_y = [], []  # ѵ����
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :2]
        # y = normalized_train_data[i:i + time_step, :0, np.newaxis]
        y = normalized_train_data[i:i + time_step, :2]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    del train_y[0]
    del train_x[len(train_x)-1]
    batch_index.append((len(normalized_train_data) - time_step))
    return mean, std, batch_index, train_x, train_y

def get_test_data(time_step=1):
    test_begin=2499
    test_end = 2500
    data_test=data[test_begin:test_end]
    # mean=np.mean(data_test,axis=0)
    # std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #��׼��
    size=(len(normalized_test_data)+time_step-1)//time_step  #��size��sample
    test_x=[]
    #�ӵ���time_step�����ݿ�ʼԤ��
    x = normalized_test_data[:, :2]
    test_x.append(x.tolist())
    print(x, '\n', test_x)
    return mean,std,test_x

#�������������������������������������������������������������������������������������
#����㡢�����Ȩ�ء�ƫ��

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,2]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[2,]))
       }

#�������������������������������������������������������������������������������������
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #��Ҫ��tensorת��2ά���м��㣬�����Ľ����Ϊ���ز������
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #��tensorת��3ά����Ϊlstm cell������
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=tf.AUTO_REUSE)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn�Ǽ�¼lstmÿ������ڵ�Ľ����final_states�����һ��cell�Ľ��
    output=tf.reshape(output_rnn,[-1, 1, rnn_unit]) #��Ϊ����������
    w_out=weights['out']
    b_out=biases['out']
    output_1=tf.reshape(output,[-1,10])
    # print(output_1, '\n', w_out, '\n', b_out)
    pred = tf.matmul(output_1,w_out)+b_out
    pred_1 = tf.reshape(pred, [-1, 1, 2])
    # print(pred_1)
    return pred_1,final_states

def train_lstm(batch_size=20,time_step=1):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, batch_index,train_x,train_y=get_train_data(batch_size,time_step)
    # print(len(train_x))
    # print(len(train_y))
    # print(train_x*std+mean)
    # print(np.array(train_y)*std[0]+mean[0])
    pred,_=lstm(X)
    #��ʧ����
    loss=tf.reduce_mean(tf.square(pred-Y))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint('/Users/liyangyang/Downloads/output/apple_stock2_pred')
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, module_file)
        #�ظ�ѵ��2000��
        for i in range(100):
            for step in range(len(batch_index)-1):
                # print('tx',train_x[batch_index[step]:batch_index[step+1]])
                # print('ty',train_y[batch_index[step]:batch_index[step+1]])
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)

            if i % 50==0:
                print("����ģ�ͣ�",saver.save(sess,'/Users/liyangyang/Downloads/output/apple_stock2_pred/stock.model',global_step=i))

def prediction(time_step=1):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x=get_test_data(time_step)
    # print(test_x * std + mean)
    # print(np.array(test_y) * std[0] + mean[0])
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #�����ָ�
        module_file = tf.train.latest_checkpoint('/Users/liyangyang/Downloads/apple_stock_pred')
        saver.restore(sess, module_file)
        test_predict=[]
        test_predict.append(test_x[0][0][0])
        for step in range(19):
            print(test_x)
            prob=sess.run(pred,feed_dict={X:[test_x[0]]})
            predict=prob.reshape((-1))
            # print('prob', prob)
            # print('predict', predict)
            test_predict.append(predict[0])
            test_x = prob.tolist()
        # test_y=np.array(test_y)*std[0]+mean[0]
        print(test_predict)
        test_predict=np.array(test_predict)*std[0]+mean[0]
        # acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) #accΪ���Լ�ƫ��
        plt.figure()
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.plot(list(range(len(test_predict))), test_predict, color='b', ls=':')
        plt.show()

train_lstm()
prediction()