from data_processing import *

def test_load_data():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_data = load_data(filenames)
    assert len(mock_data) == 20

def test_process_data():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_data = load_data(filenames)
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    index_to_continuous = list(
        set(mock_data.columns.values)-set(index_to_category))
    data, labels = process_data(
        mock_data, index_to_category, index_to_continuous)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        data, labels)
    assert data.shape == (20, 49)
    assert labels.shape == (20, )
    assert len(train_data) == 15
    assert len(train_labels) == 15
    assert len(dev_data) == 5
    assert len(dev_labels) == 5

def test_convert_to_onehot():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_data = load_data(filenames)
    index_to_category = ['protocol_type', 'service', 'flag', 'class']
    categorical_dataset = mock_data[index_to_category]
    datasets = convert_to_onehot(categorical_dataset)
    assert datasets.shape[1] == 17

if __name__=="__main__":
    test_load_data()
    test_process_data()
    test_convert_to_onehot()
    print("test is all passed")
