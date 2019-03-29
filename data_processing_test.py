from data_processing import *

def test_load_data():
    filenames = ['./mock_data.txt', './mock_data.txt']
    mock_data = load_data(filenames)
    assert len(mock_data) == 20

if __name__=="__main__":
    test_load_data()
    print("test is all passed")
