import tensorflow as tf
#声明变量w1,w2，这里还通过seed参数设定了随机种子
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))
#暂时讲输入的特征向量定义为一个常量。注意这里x是一个1*2的矩阵
x = tf.constant([[0.7,0.9]])
#定义前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a,w2)

#构建计算图到本地，然后以tensorboard查看计算图
writer = tf.summary.FileWriter(r'E:\TensorBoard',tf.get_default_graph())
writer.close()

#开启会话，来执行计算
sess =tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()
