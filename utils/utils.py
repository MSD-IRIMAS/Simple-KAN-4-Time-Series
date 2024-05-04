import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


def load_data(file_name):
    folder_path = "/home/myuser/ucr_archive/"
    folder_path += file_name + "/"

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)
    
    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)
    
    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    return xtrain, ytrain, xtest, ytest


def znormalisation(x):
    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):
    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
