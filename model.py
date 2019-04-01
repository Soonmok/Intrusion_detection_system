import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, X_dense, hidden_size):
        features = self.encode(X_dense, hidden_size)
        self.X_dense_reconstructed = self.decode(features, X_dense.shape[1])

    def encode(self, x_input, hidden_size):
        layer_1 = tf.layers.dense(
            inputs=x_input,
            units=1024,
            activation=tf.nn.sigmoid,
            name="encode1")
        features = tf.layers.dense(
            inputs=layer_1,
            units=hidden_size,
            activation=tf.nn.sigmoid,
            name="encode2")
        with tf.variable_scope('encode1', reuse=True):
            self.w_encoder_1 = tf.get_variable('kernel')
        with tf.variable_scope('encode2', reuse=True):
            self.w_encoder_2 = tf.get_variable('kernel')
        return features

    def decode(self, features, reconstructed_size):
        layer_1 = tf.layers.dense(
            inputs=features,
            units=1024,
            name="decode1")
        X_dense_reconstructed = tf.layers.dense(
            inputs=layer_1,
            units=reconstructed_size,
            name="decode2")
        return X_dense_reconstructed
