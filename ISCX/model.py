import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, X_dense, hidden_size, num_classes, config):
        encoded_features = self.encode(X_dense, hidden_size, config)
        self.X_reconstructed = self.decode(encoded_features, X_dense.shape[1])
        self.logits = self.classify(encoded_features, num_classes)

    def encode(self, x_input, hidden_size, config):
        dropout_layer = tf.layers.dropout(
            x_input,
            rate=config.STL_dropout_rate)
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

    def classify(self, encoded_features, num_classes):
        logits = tf.layers.dense(
            inputs=encoded_features,
            units=num_classes,
            activation=tf.nn.sigmoid,
            name="logits")
        return logits

