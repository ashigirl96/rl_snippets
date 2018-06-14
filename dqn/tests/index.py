import tensorflow as tf

REF = tf.Variable(tf.zeros([128]), False)

def main(_):
    ref = REF.assign(tf.reshape(tf.random_uniform([32, 4]), [-1]))
    indices = tf.range(32) * 4
    updates = tf.zeros([32])
    update = tf.scatter_nd_update(ref, indices[:, None], updates)
    update = tf.reshape(update, [32, 4])
    update = tf.convert_to_tensor(update)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        print(sess.run(updates))
        print(sess.run(update))


if __name__ == '__main__':
    tf.app.run()