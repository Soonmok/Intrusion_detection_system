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
        '--data_path', type=str,
        default='./dataset/Thursday-WorkingHours.pcap_Flow.csv')
    args.add_argument('--STL_learning_rate', type=float, default=0.0001)
    args.add_argument('--STL_epoch', type=int, default=100)
    args.add_argument('--STL_patient_cnt', type=int, default=200)
    args.add_argument('--STL_dropout_rate', type=float, default=0.5)
    args.add_argument('--min_delta', type=float, default=0.000001)
    args.add_argument('--classify_learning_rate', type=float, default=0.0001)
    args.add_argument('--classify_epoch', type=int, default=200)
    config = args.parse_args()

    datasets = load_data([config.data_path])
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    index_to_continuous = list(set(datasets.columns.values)-set(index_to_category))
    unnecessary_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp',
                       'Label']
    data, labels=  process_data(datasets, unnecessary_cols)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        data, labels)
    num_features = train_data.shape[1]
    num_classes = train_labels.shape[0]
    print("train / dev  --> {} / {}".format(len(train_data), len(dev_data)))
    print("total features -> {}".format(num_features))
    
    """------------model part------------"""
    input_x = tf.placeholder(
        tf.float32, shape=(config.batch_size, num_features))
    input_y = tf.placeholder(
        tf.int64, shape=(config.batch_size, num_classes))
    STL_global_step = tf.Variable(0, name="global_step")

    # STL part
    ae_model = AutoEncoder(input_x, config.hidden_size, num_classes, config)
    reconstruction_cost = tf.losses.mean_squared_error(
       input_x, ae_model.X_reconstructed)

    total_loss = reconstruction_cost 

    train_op = tf.train.AdamOptimizer(
        config.STL_learning_rate).minimize(total_loss, global_step=STL_global_step)
     
    # Classification part
    classify_global_step = tf.Variable(0, name="classify_global_step")
    classification_cost = tf.losses.softmax_cross_entropy(
        input_y, ae_model.logits)
    prediction = tf.argmax(ae_model.logits, 1, name="prediction")
    classification_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(prediction, tf.argmax(input_y, 1)), "float"), name="accuracy")
    softmax_classifier_op = tf.train.GradientDescentOptimizer(
        config.classify_learning_rate).minimize(
            classification_cost, global_step=classify_global_step)

    # Init Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    train_dataset = get_train_dataset(data, labels, config.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()
    train_batch = train_iterator.get_next()

    # STL training part
    def train_step(batch_data):
        train_batch_data, _ = sess.run(batch_data)
        train_batch_data_norm = tf.keras.utils.normalize(train_batch_data)
        feed_dict = {input_x: train_batch_data_norm}
        x_data, _, cost, step= sess.run(
            [input_x, train_op, total_loss, STL_global_step], feed_dict)
        return cost, step

    epoch = 1
    patience_cnt = 0
    cur_cost = 0
    while True:
        try:
            prev_cost = cur_cost
            cur_cost, step = train_step(train_batch)

            if step % 100 == 0:
                print("step {}, cost {} ".format(step, cur_cost))

            if prev_cost - cur_cost > config.min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt > config.STL_patient_cnt:
                print("early stopping")
                break
                
        except ValueError: 
            print("==============================")
            epoch += 1
            if epoch > config.STL_epoch:
                break

    # Classification part
    def train_classify_step(batch_data):
        train_batch_data, train_batch_label = sess.run(batch_data)
        feed_dict = {input_x: train_batch_data,
                     input_y: train_batch_label}
        _, cost, acc, step = sess.run(
            [softmax_classifier_op, classification_cost,
             classification_accuracy, classify_global_step], feed_dict)
        return cost, acc, step
    
    saver = tf.train.Saver()
    epoch = 1
    while True:
        try:
            cost, acc, step = train_classify_step(train_batch)
            if step % 100 == 0:
                print("classification step {}, classification cost {}"
                      "  acc {}".format(step, cost, acc))
            if step % 1000 == 0:
                print("saving model ...")
                path = saver.save(sess, "./models/checkpoint",
                                  global_step=classify_global_step)
                print("saved model checkpoint to {}\n".format(path))
        except ValueError:
            print("==============================")
            epoch += 1
            if epoch > config.classify_epoch:
                break

