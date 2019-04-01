from data_processing import *

def test_load_data():
    filenames = ['./mock_data.txt', './mock_data.txt']
    mock_data = load_data(filenames)
    assert len(mock_data) == 20

def test_process_data():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_datasets = load_data(filenames)
    train_data, train_labels, _, _ = process_data(mock_datasets)
    assert train_data.shape == (15, 41)
    assert train_labels.shape == (15, )

if __name__=="__main__":
    test_load_data()
    test_process_data()
    print("test is all passed")
