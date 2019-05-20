from data_processing import *

def test_load_data():
    filenames = ['./mock_data.csv', './mock_data2.csv']
    mock_data = load_data(filenames)
    assert len(mock_data) == 20

def test_process_data():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_data = load_data(filenames)
    data, labels = process_data(mock_data)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        data, labels)
    assert data.shape == (20, 83)
    assert labels.shape == (20, 1)
    assert len(train_data) == 15
    assert len(train_labels) == 15
    assert len(dev_data) == 5
    assert len(dev_labels) == 5

if __name__=="__main__":
    test_load_data()
    test_process_data()
    print("test is all passed")