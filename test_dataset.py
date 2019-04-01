import pandas as pd

if __name__=="__main__":
    filename = './dataset/train_data/KDDTrain+.txt'
    datasets = pd.read_csv(filename, sep=',', header=None)
    labels = datasets.iloc[:, -2]
    labels = set(labels)
    print(len(labels))
