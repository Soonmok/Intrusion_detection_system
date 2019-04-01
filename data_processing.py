import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filenames):
    datasets = []
    for filename in filenames:
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
        dataset = pd.read_csv(filename, sep=',', header=None, names=names)
        datasets.append(dataset)
    datasets = pd.concat(datasets, axis=0, ignore_index=True)
    return datasets

def process_data(datasets, index_to_category, index_to_continuous):
    label_encoder = LabelEncoder()
    encode_label(datasets, index_to_category)

    indices = range(len(datasets))
    train_indices, dev_indices = train_test_split(indices, shuffle=True)
    print("loading total data {} train data {}, dev_data {}".format(
        len(datasets), len(train_indices), len(dev_indices)))

    train_data = datasets.iloc[train_indices, :41].values
    train_labels = datasets.iloc[train_indices, 41].values
    dev_data = datasets.iloc[dev_indices, :41].values
    dev_labels = datasets.iloc[dev_indices, 41].values
    return train_data, train_labels, dev_data, dev_labels
    
def encode_label(data, index_to_category):
    for index in index_to_category:
        le = LabelEncoder().fit(data[index])
        data[index] = le.transform(data[index])
        data[index] = pd.get_dummies(data[index])



