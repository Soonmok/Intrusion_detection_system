# Intrusion_detection_system
implement IDS using deep learning

# deep_IDS

## Implement A Deep Learning Approach for Network Intrusion Detection System paper

## Requirements

numpy

sklearn

pandas

tensorflow-1.13.1


## Dataset
Download the zipfile from this url
https://www.unb.ca/cic/datasets/nsl.html

unzip the file into `Intrusion_detection_system/deep_IDS/dataset/train_data`

edit the file `KDDTrain+_20Percent.arff`   (remove @attributes lines (33 lines))

rename `KDDTrain+_20Percent.arf` into `KDDTrain_binary.txt`

`mv KDDTrain+_20Percent.arff KDDTrain_binary.txt`

## How to run
`
python main.py
`

if you want to specify the path of dataset 

` python main.py --data_path="the/path/to/dataset`
## references 
A Deep Learning Approach for Network Intrusion Detection
System

http://www.covert.io/research-papers/deep-learning-security/A%20Deep%20Learning%20Approach%20for%20Network%20Intrusion%20Detection%20System.pdf
