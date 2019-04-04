import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, X_dense, hidden_size):
        encoded_features = self.encode(X_dense, hidden_size)
        self.X_reconstructed = self.decode(encoded_features, X_dense.shape[1])

    def encode(self, x_input, hidden_size):
        self.normalized = tf.nn.sigmoid(x_input)
        dropout_layer = tf.layers.dropout(
            self.normalized,
            rate=0.5)
        features = tf.layers.dense(
            inputs=dropout_layer,
            units=hidden_size,
            activation=tf.nn.sigmoid,
            name="encode")
        with tf.variable_scope('encode', reuse=True):
            self.w_encoder = tf.get_variable('kernel')
        return features

    def decode(self, features, reconstructed_size):
        X_reconstructed = tf.layers.dense(
            inputs=features,
            units=reconstructed_size,
            activation=tf.nn.sigmoid,
            name="decode")
        with tf.variable_scope('decode', reuse=True):
            self.w_decoder = tf.get_variable('kernel')
        return X_reconstructed
