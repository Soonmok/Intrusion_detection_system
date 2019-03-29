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
    
