import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(filenames):
    datasets = []
#     names = ['FlowID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port', 'Protocol',
#             'Timestamp', 'Flow_Duration', 'Tot_fwd_pkts', 'Tot_bwd_pkts',
#             'TotLen_fwd_pkts', 'Totlen_bwd_pkts', 'Fwd_Pkt_len_Max',
#              'Fwd_Pkt_len_Min', 'Fwd_Pkt_len_mean', 'Fwd_Pkt_len_std'.
#             'Bwd_Pkt_len_Max', 'Bwd_Pkt_len_Min', 'Bwd_Pkt_len_Mean', 
#             'Bwd_Pkt_len_Std', 'Flow_Bytes/s', 'Flow_Pkts/s', 'Flow_IAT_Mean',
#             'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Tot',
#             'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
#             'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max',
#             'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags',
#             'Bwd_URG_Flags', 'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_Pkts/s',
#             'Bwd_Pkts/s', 'Ptk_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean',
#              'Pkt_Len_Std', 'Pkt_Len_Var', 'Fin_Flag_Cnt', 'Syn_Flag_Cnt', 
#             'Rst_Flag_Cnt', 'PSH_Flag_Cnt', 'Ack_Flag_Cnt', 'Ugr_Flag_Cnt',
#             'Cwe_Flag_Cnt', 'Ece_Flag_Cnt', 'Down/Up_Ratio', 'Pkt_Size_Avg',
#             'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg', 'Fwd_Byts/b_Avg',
#             'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Bytes/b_Avg',
#              'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Subflow_Fwd_Pkts',
#             'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_']
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

def process_data(data):
    np_datasets = data[index_to_continuous].values
    labels = pd.DataFrame({'labels':categorical_dataset.pop('Label')})
    return np_datasets, labels 

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
