import tensorflow as tf
# tf,constant是一个计算过程，计算结果保存为一个张量，保存在变量a中
# a =tf.constant([1.0,2.0],name='a')
# b =tf.constant([2.0,3.0],name='b')
# result = tf.add(a,b,name='add')
# print(result)
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a + b
print(tf.Session().run(result))