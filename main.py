import tensorflow as tf
import model
from data_processing import *

if __name__=="__main__":
    datasets = load_data(['./dataset/train_data/KDDTrain+.txt'])
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    index_to_continuous = list(set(datasets.columns.values)-set(index_to_category))
    data, labels=  process_data(
        datasets, index_to_category, index_to_continuous)

