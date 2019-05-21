import tensorflow as tf
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(dirname):
    datasets = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        dataset = pd.read_csv(full_filename, sep=',')
        datasets.append(dataset)
    datasets = pd.concat(datasets, axis=0, ignore_index=True)
    datasets = datasets.reindex(np.random.permutation(datasets.index))
    datasets = clean_data(datasets)
    return datasets

def clean_data(dataset):
    dataset['Flow Byts/s'] = pd.to_numeric(dataset['Flow Byts/s'], errors='coerce')
    dataset['Flow Pkts/s'] = pd.to_numeric(dataset['Flow Pkts/s'], errors='coerce')
    cleaned_dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]
    return cleaned_dataset

def convert_to_onehot(data):
    enc = OneHotEncoder(categories='auto')
    le = LabelEncoder()
    categorical_dataset = data.apply(le.fit_transform)
    enc.fit(categorical_dataset)
    onehot_dataset = enc.transform(categorical_dataset).toarray()
    return onehot_dataset

def process_data(data, unnecessary_cols):
    columns = [x for x in list(data.columns) if x not in unnecessary_cols]
    np_datasets = data[columns].values
    np_labels = convert_to_onehot(pd.DataFrame({'Label':data['Label']}))
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

