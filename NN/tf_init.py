import tensorflow as tf


def hands_on():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x*x*y + y + 2

    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
    return result


def hands_on_another():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x*x*y + y + 2
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        result = f.eval()
    return result


if __name__ == '__main__':
    print(hands_on_another())
