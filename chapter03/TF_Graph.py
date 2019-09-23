import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v =tf.get_variable(
        "v",shape=[1],initializer=tf.zeros_initializer #v值为0
    )

g2 =tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        "v",shape=[1],initializer=tf.ones_initializer #v值为1
    )

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        writer = tf.summary.FileWriter("E://TensorBoard//test_0", sess.graph)
        print(sess.run(tf.get_variable('v')))

writer.close()
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        writer = tf.summary.FileWriter("E://TensorBoard//test_1", sess.graph)
        print(sess.run(tf.get_variable('v')))

writer.close()

