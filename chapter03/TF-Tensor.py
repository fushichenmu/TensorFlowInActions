import tensorflow as tf
# tf,constant是一个计算过程，计算结果保存为一个张量，保存在变量a中
# a =tf.constant([1.0,2.0],name='a')
# b =tf.constant([2.0,3.0],name='b')
# result = tf.add(a,b,name='add')
# print(result)

b = tf.constant([2.0,3.0],name='b')
a = tf.constant([1.0,2.0],name='cccc')
result = a + b
result2 = result*2

print(tf.Session().run(result))
print(tf.Session().run(result2))


g1 = tf.Graph()
with g1.as_default():
    a = tf.constant([1.0, 2.0], name='ddddd')
    b = tf.constant([2.0, 3.0], name='b')
    result = a + b
    print(result)
print(a)