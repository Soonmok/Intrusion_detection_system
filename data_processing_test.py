from data_processing import *

def test_load_data():
    filenames = ['./mock_data.txt', './mock_data.txt']
    mock_data = load_data(filenames)
    assert len(mock_data) == 20

def process_data_test():
    filenames = ['./mock_data.txt', './mock_data2.txt']
    mock_datasets = load_data(filenames)
    data, labels = process_data(mock_datasets)
    assert data.shape == (20, 41)
    assert labels.shape == (20, )

if __name__=="__main__":
    test_load_data()
    process_data_test()
    print("test is all passed")
