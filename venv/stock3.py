#coding=gbk
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#���峣��
rnn_unit=10       #hidden layer units
input_size=3
output_size=1
lr=0.0006         #ѧϰ��

f=open('/Users/liyangyang/Downloads/fund_data_2.csv')
df=pd.read_csv(f)     #�����Ʊ����
data_0=df.iloc[:1030,1:4].values   #ȡ��2-4��
data = data_0[-1::-1, :] #����
print (data)


#��ȡѵ����
def get_train_data(batch_size=50,time_step=10,train_begin=0,train_end=800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    mean = np.mean(data_train, axis=0)
    std = np.std(data_train, axis=0)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #��׼��
    train_x,train_y=[],[]   #ѵ����
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:3]
       y=normalized_train_data[i:i+time_step,2,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return mean, std, batch_index,train_x,train_y

#����������������������ȡ���Լ���������������������
def get_test_data(time_step=5,test_begin=800):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #��׼��
    size=(len(normalized_test_data)+time_step-1)//time_step  #��size��sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:3]
       y=normalized_test_data[i*time_step:(i+1)*time_step,2]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(size)*time_step:,:3]).tolist())
    test_y.extend((normalized_test_data[(size)*time_step:,2]).tolist())
    return mean,std,test_x,test_y

#�������������������������������������������������������������������������������������
#����㡢�����Ȩ�ء�ƫ��

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
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
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #��Ϊ����������
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#������������������������������������ѵ��ģ�͡�����������������������������������
def train_lstm(batch_size=50,time_step=5,train_begin=0,train_end=800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #��ʧ����
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint('/Users/liyangyang/Downloads/fund2')
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, module_file)
        #�ظ�ѵ��2000��
        for i in range(500):
            for step in range(len(batch_index)-1):
                if i == 99:
                    train_pred = []

                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})

                if i == 99:
                    prob_0 = sess.run(pred,feed_dict={X:[train_x[step]]})
                    print(prob_0)
                    predict = prob_0.reshape((-1))
                    train_pred.extend(predict)

            print(i,loss_)

            # if i == 99:
            #     train_end_predict = np.array(train_pred)*std[2] + mean[2]
            #     train_y_0 = np.array(train_y) * std[2] + mean[2]
            #     plt.figure()
            #     plt.plot(list(range(len(train_y_0))), train_y_0, color='r')
            #     plt.plot(list(range(len(train_end_predict))), train_end_predict, color='b', ls=':')
            #     plt.show()
            if i % 50==0:
                print("����ģ�ͣ�",saver.save(sess,'/Users/liyangyang/Downloads/fund2/stock2.model',global_step=i))



#��������������������������������Ԥ��ģ�͡���������������������������������������
def prediction(time_step=5):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #�����ָ�
        module_file = tf.train.latest_checkpoint('/Users/liyangyang/Downloads/fund2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[2]+mean[2]
        test_predict=np.array(test_predict)*std[2]+mean[2]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) #accΪ���Լ�ƫ��
        plt.figure()
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.plot(list(range(len(test_predict))), test_predict, color='b', ls=':')
        plt.show()

train_lstm()
prediction()