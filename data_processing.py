import tensorflow as tf
import numpy as np
import pandas as pd

def load_data(filenames):
    datasets = []
    for filename in filenames:
        dataset = pd.read_csv(filename, sep=',', header=None)
        datasets.append(dataset)
    datasets = pd.concat(datasets, axis=0, ignore_index=True)
    return datasets

def process_data(datasets):
    index_to_category = [1, 2, 3, 6, 11, 20, 21, 41]
    datasets[index_to_category] = datasets[index_to_category].astype('category')
    data = datasets.iloc[:, :41]
    labels = datasets.iloc[:, 41]
    return data, labels
    
