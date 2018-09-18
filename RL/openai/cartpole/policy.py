import tensorflow as tf


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


def network(obs, n_inputs, n_hidden, n_outputs, learning_rate):
    initializer = tf.contrib.layers.variance_scaling_initializer()

    x = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(x, n_hidden, activation=tf.nn.elu,
                             kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs,
                             kernel_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
    return action