import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(filenames):
    datasets = []
    names = [ 'Flow ID','Src IP','Src Port','Dst IP','Dst Port','Protocol',
             'Timestamp','Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts',
             'TotLen Fwd Pkts','TotLen Bwd Pkts','Fwd Pkt Len Max',
             'Fwd Pkt Len Min','Fwd Pkt Len Mean','Fwd Pkt Len Std',
             'Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd Pkt Len Mean',
             'Bwd Pkt Len Std','Flow Byts/s','Flow Pkts/s','Flow IAT Mean',
             'Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Tot',
             'Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Tot',
             'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags',
             'Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Len',
             'Bwd Header Len','Fwd Pkts/s','Bwd Pkts/s','Pkt Len Min',
             'Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt',
             'SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt',
             'URG Flag Cnt','CWE Flag Count','ECE Flag Cnt','Down/Up Ratio',
             'Pkt Size Avg','Fwd Seg Size Avg','Bwd Seg Size Avg','Fwd Byts/b Avg',
             'Fwd Pkts/b Avg','Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg',
             'Bwd Blk Rate Avg','Subflow Fwd Pkts','Subflow Fwd Byts',
             'Subflow Bwd Pkts','Subflow Bwd Byts','Init Fwd Win Byts',
             'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min',
             'Active Mean','Active Std','Active Max','Active Min,Idle Mean',
             'Idle Std,Idle Max','Idle Min','Label']
    for filename in filenames:
        dataset = pd.read_csv(filename, sep=',')
        datasets.append(dataset)
    datasets = pd.concat(datasets, axis=0, ignore_index=True)
    return datasets

def process_data(data):
    np_datasets = data.values[:, :-1]
    np_labels = data.values[:, -1]
    return np_datasets, np_labels 

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
