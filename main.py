import tensorflow as tf
from model import *
from data_processing import *
import argparse

if __name__=="__main__":

    # setting parameters
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--hidden_size', type=int, default=200)
    args.add_argument(
        '--data_path', type=str, default='./dataset/train_data/KDDTrain+.txt')
    args.add_argument('--learning_rate', type=float, default=0.001)

    datasets = load_data([config.data_path])
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    index_to_continuous = list(set(datasets.columns.values)-set(index_to_category))
    data, labels=  process_data(
        datasets, index_to_category, index_to_continuous)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        datasets, labels)
    num_features = train_data.shape[0]
    
    """------------model part------------"""
    input_x = tf.placeholder(
        tf.float32, shape=(config.batch_size, num_features), dtype=np.int64)

    input_y = tf.placeholder(
        tf.float32, shape=(config.batch_size, ), dtype=np.int64)
    global_step = tf.Variable(0, name="global_step")

    ae_model = AutoEncoder(input_x, config.hidden_size)

    reconstruction_cost = tf.losses.mean_squared_error(
        input_x, ae_model.X_dense_reconstructed)
    regulation_cost = tf.reduce_sum(model.w_encoder_1 ** 2) + \
            tf.reduce_sum(model.w_encoder_2 ** 2) + \
            tf.reduce_sum(model.w_decoder_1 ** 2) + \
            tf.reduce_sum(model.w_decoder_2 ** 2)

    total_loss = reconstruction_cost + regulation_cost * 0.5

    train_op = tf.train.AdamOptimizer(
        config.learning_rate).minimize(total_loss, global_step=global_step)
     
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)



