import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 导入数据
data = pd.read_csv('/Users/liyangyang/Downloads/data_stocks.csv')
# 移除日期列
data = data.drop(['DATE'], 1)
# 数据集的维度
n = data.shape[0]
p = data.shape[1]
# 将数据集转化为numpy数组
data = data.values

# 划分训练集和测试集
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# 数据缩放
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# 构建 X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# 定义 a 和 b 为占位符
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)

# 定义加法运算
c = tf.add(a, b)

# 初始化图
graph = tf.Session()

# 运行图
graph.run(c, feed_dict={a: 5, b: 4})

# n_stocks = 500
# #训练集的股票数量
n_stocks = X_train.shape[1]

# Session
net = tf.InteractiveSession()

# 占位符
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# 定义初始化器
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# 模型结构参数
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# 第一层 : 隐藏层权重和偏置变量
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# 第二层 : 隐藏层权重和偏置变量
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# 第三层: 隐藏层权重和偏置变量
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# 第四层: 隐藏层权重和偏置变量
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# 输出层: 输出权重和偏置变量
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# 隐藏层
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# 输出层 (必须经过转置)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# 损失函数
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# 优化器
opt = tf.train.AdamOptimizer().minimize(mse)

# 运行初始化器
net.run(tf.global_variables_initializer())

# 设定用于展示交互的图表
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# 设定 epochs 数和每批次的数据量
epochs = 10
batch_size = 256
mse_train = []
mse_test = []

for e in range(epochs):

    # 打乱训练集
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch 训练
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # 在当前batch上运行优化器
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # 展示进度
        if np.mod(i, 5) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = '/Users/liyangyang/Downloads/img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)
# 展示训练结束时最终的MSE
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)