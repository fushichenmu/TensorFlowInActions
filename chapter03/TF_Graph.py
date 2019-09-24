# 3.1 计算图的使用
#一般别名tf
import tensorflow as tf

'''
默认计算图的使用
    TensorFlow程序中，系统会维护一个默认的计算图，
    可以通过tf.get_default_graph()方式得到当前默认的计算图
'''
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
result = a + b
print(a.graph is tf.get_default_graph())


'''
自定义计算图的使用
'''
#创建自定义计算图
g1 = tf.Graph()
#在计算图g1中定义变量“v”,并设置初始值为0
with g1.as_default():
    v = tf.get_variable(
        "v",shape=[1],initializer=tf.zeros_initializer
    )

g2 = tf.Graph()
# 在计算图g2中定义变量“v”,并设置初始值为1
with g2.as_default():
    v = tf.get_variable(
        "v",shape=[1],initializer=tf.ones_initializer
    )
#在计算图g1中读取变量v的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run() #变量初始化
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v"))) #获取变量v的值
#在计算图g2中读取变量v的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run() #变量初始化
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v"))) #获取变量v的值

#使用GPU，将在第12章详述
g = tf.Graph()
with g.device('/gpu:0'):
    result = a+b
