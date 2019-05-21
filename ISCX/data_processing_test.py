from data_processing import *

def test_load_data():
    mock_data = load_data('./test_dataset')
    assert len(mock_data) == 20

def test_infinity():
    mock_data = load_data('./test_dataset')

def test_process_data():
    unnecessary_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp',
                       'Label']
    mock_data = load_data('./test_dataset')
    data, labels = process_data(mock_data, unnecessary_cols)
    train_data, train_labels, dev_data, dev_labels = devide_train_dev(
        data, labels)
    assert data.shape == (20, 78)
    assert labels.shape == (20, 1)
    assert len(train_data) == 15
    assert len(train_labels) == 15
    assert len(dev_data) == 5
    assert len(dev_labels) == 5

if __name__=="__main__":
    test_load_data()
    test_process_data()
    test_infinity()
    print("test is all passed")
