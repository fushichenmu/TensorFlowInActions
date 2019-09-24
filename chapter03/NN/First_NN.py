import tensorflow as tf

#变量的使用
a = tf.Variable(tf.random_normal([2, 3], stddev=1, mean=0)) #标准正太
b = tf.Variable(tf.truncated_normal([2, 3], stddev=1, mean=0))#正态分布，但如果随机出来的值偏离平均值超过2个标准差，那么这个数将被重新随机。相当于处理了异常值的情况。
c = tf.Variable(tf.random_gamma([2, 3],alpha=0.1))#Gamma分布
d = tf.Variable(tf.random_uniform([2, 3],minval=0.5,maxval=1.0))#均匀分布

with tf.Session().as_default() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))

#常量的使用
e = tf.Variable(tf.zeros([2, 3], dtype=tf.int32)) #全0矩阵
f = tf.Variable(tf.ones([2, 3], dtype=tf.int32)) #全1矩阵
g = tf.Variable(tf.fill([2, 3],1)) #全fill值矩阵
h = tf.constant([2, 3], dtype=tf.int32) #直接指定常量 ，注意了constant不是大写！

with tf.Session().as_default() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    print(sess.run(e))
    print(sess.run(f))
    print(sess.run(g))
    print(sess.run(h))