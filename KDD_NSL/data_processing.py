import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(filenames):
    datasets = []
    names = ['duration', 'protocol_type', 'service', 'flag', 'src_byte',
                  'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                  'num_failed_logins', 'logged_in', 'num_compromised',
                   'root_shell', 'su_attempted', 'num_root',
                   'num_file_creations', 'num_shells', 
                  'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                  'is_guest_login', 'count', 'srv_count', 'serror_rate',
                  'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                  'dst_host_count', 'dst_host_srv_count',
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                  'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                  'class', 'dummy']
    for filename in filenames:
        dataset = pd.read_csv(filename, sep=',', header=None, names=names)
        datasets.append(dataset)
    datasets = pd.concat(datasets, axis=0, ignore_index=True)
    # delete unnecessary data
    datasets.pop('land')
    datasets.pop('dummy')
    return datasets

def convert_to_onehot(data):
    enc = OneHotEncoder(categories='auto')
    le = LabelEncoder()
    categorical_dataset = data.apply(le.fit_transform)
    enc.fit(categorical_dataset)
    onehot_dataset = enc.transform(categorical_dataset).toarray()
    return onehot_dataset

def process_data(data, index_to_category, index_to_continuous):
    continuous_dataset = data[index_to_continuous].values
    categorical_dataset = data[index_to_category]
    labels = pd.DataFrame({'labels':categorical_dataset.pop('class')})
    categorical_dataset = convert_to_onehot(categorical_dataset)
    categorical_labels = convert_to_onehot(labels)
    np_datasets = np.concatenate(
        (continuous_dataset, categorical_dataset), axis=1) 
    return np_datasets, categorical_labels

def devide_train_dev(datasets, labels):
    indices = range(len(datasets))
    train_indices, dev_indices = train_test_split(indices, shuffle=True)
    print("loading total data {} train data {}, dev_data {}".format(
        len(datasets), len(train_indices), len(dev_indices)))
    train_data = datasets[train_indices]
    train_labels = labels[train_indices]
    dev_data = datasets[dev_indices]
    dev_labels = labels[dev_indices]
    return train_data, train_labels, dev_data, dev_labels

def generator(datasets):
    for data, label in datasets:
        yield data, label
 
def get_train_dataset(data, labels, batch_size):
    datasets = list(zip(data, labels))
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: generator(datasets), (tf.float32, tf.int64))
    return tf_dataset.batch(batch_size).repeat()
