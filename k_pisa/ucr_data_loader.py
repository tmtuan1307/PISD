from os import path
from numpy import genfromtxt
import scipy.stats as stats
import numpy as np


def load_dataset(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter='\t')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter='\t')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    return train_data, train_labels, test_data, test_labels


def load_dataset_zscore(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter='\t')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter='\t')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    z_train_data = np.asarray([stats.zscore(data) for data in train_data])
    z_test_data = np.asarray([stats.zscore(data) for data in test_data])

    return z_train_data, train_labels, z_test_data, test_labels


def load_dataset_varylen(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter='\t')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter='\t')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    new_train_data = [data[np.isnan(data) == False] for data in train_data]
    new_test_data = [data[np.isnan(data) == False] for data in test_data]

    return new_train_data, train_labels, new_test_data, test_labels


def sort_data_by_error_list(data, error_list):
    error_data = []
    correct_data = []
    for i in range(len(error_list)):
        if error_list[i]:
            correct_data.append(data[i])
        else:
            error_data.append(data[i])
    new_data = error_data + correct_data
    if len(error_list) < len(data):
        for i in range(len(error_list), len(data)):
            new_data.append(data[i])

    return new_data

