import tensorflow as tf

#创建一个会话
# sess = tf.Session()
#使用这个创建好的会话来得到关心的运算的结果。比如可以调用sess.run(result)
# sess.run(...)
#关闭会话使得本次运行中使用到的资源可以被释放
# sess.close()


#创建一个会话，并通过Python中的上下文管理器来管理这个会话
# with tf.Session() as sess:
    #使用创建好的会话来计算关心的结果
    # sess.run(...)
#不需要在调用‘Session.close()’函数来显示的释放资源了
#当上下文退出是，会话关闭和资源释放也同时完成了


sess = tf.Session()
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
result = a + b
# with sess.as_default():
#     print(a.eval()) #[1. 2.]
#     print(b.eval()) #[1. 2.]
#     print(result.eval()) #[2. 4.]
print(sess.run(result))
print(result.eval(session= sess))


a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
result = a + b
session = tf.InteractiveSession()
print(result.eval())
session.close()

tf_config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
interactive_session = tf.InteractiveSession(config=tf_config_proto)
print(result.eval())
interactive_session.close()
# tf_session = tf.Session(config=tf_config_proto)
