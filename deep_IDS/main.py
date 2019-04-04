import tensorflow as tf
from model import *
from data_processing import *
import argparse
import sys
np.set_printoptions(threshold=sys.maxsize)
if __name__=="__main__":

    # setting parameters
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--hidden_size', type=int, default=32)
    args.add_argument(
        '--data_path', type=str, default='./dataset/train_data/KDDTrain+.txt')
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=20000)
    config = args.parse_args()

    datasets = load_data([config.data_path])
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    index_to_continuous = list(set(datasets.columns.values)-set(index_to_category))
    data, labels=  process_data(
        datasets, index_to_category, index_to_continuous)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        data, labels)
    num_features = train_data.shape[1]
    
    """------------model part------------"""
    input_x = tf.placeholder(
        tf.float32, shape=(config.batch_size, num_features))

    global_step = tf.Variable(0, name="global_step")
    ae_model = AutoEncoder(input_x, config.hidden_size)
    reconstruction_cost = tf.losses.mean_squared_error(
        ae_model.normalized, ae_model.X_reconstructed)

    total_loss = reconstruction_cost 

    train_op = tf.train.AdamOptimizer(
        config.learning_rate).minimize(total_loss, global_step=global_step)
     
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    train_dataset = get_train_dataset(data, config.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()
    train_batch = train_iterator.get_next()

    def train_step(batch_data):
        train_batch_data = sess.run(batch_data)
        feed_dict = {input_x: train_batch_data}
        _, cost, step= sess.run(
            [train_op, total_loss, global_step], feed_dict)
        return cost, step

    epoch = 1
    while True:
        try:
            cost, step = train_step(train_batch)
            if step % 100 == 0:
                print("step {}, cost {} ".format(step, cost))
        except ValueError: 
            print("==============================")
            epoch += 1
            if epoch < config.epoch:
                pass
            else:
                break

