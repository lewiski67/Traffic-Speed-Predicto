# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
把数据集分割为训练集、验证集和测试集
"""

import numpy as np
import data_preprocess
from collections import namedtuple
from sklearn.model_selection import train_test_split


class DataSet(object):
    def __init__(self, speed, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._speed = speed
        self._labels = labels
        self._num_examples = speed.shape[0]
        pass

    @property
    def speed(self):
        return self._speed

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle = True):
        """
        Return the next 'batch_size' examples from this dataset
        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if start == 0 and self._epochs_completed == 0 and shuffle:
            idx = np.arange(self._num_examples)
            np.random.shuffle(idx)
            self._speed = self.speed[idx]
            self._labels = self.labels[idx]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            speed_rest_part = self._speed[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                idx0 = np.arrange(self._num_examples)
                np.random.shuffle(idx0)
                self._speed = self.speed[idx0]
                self._labels = self.labels[idx0]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            speed_new_part = self._speed[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((speed_rest_part, speed_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._speed[start:end], self._labels[start:end]

def create_data_sets():
    samples = data_preprocess.samples
    look_back = 4#意为用前多少个时间步作为输入
    interval = 0#预测之后第几个时间步时的数据
    speed, labels = [], []
    for i in range(len(samples)-look_back-interval):
        speed.append(samples[i:(i+look_back)])
        labels.append(samples[i+look_back+interval])
    return np.asarray(speed), np.asarray(labels)

def read_data_sets():
    speed, labels = create_data_sets()
    total_size = len(speed)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size
    print(total_size)
    print(train_size)
    print(val_size)
    print(test_size )
    # train_speed, test_speed, train_labels, test_labels = train_test_split(speed, labels, test_size = 0.2,random_state=0)
    train_flow = speed[:train_size]
    train_labels = labels[:train_size]

    validation_speed = speed[train_size:train_size + val_size]
    validation_labels = labels[train_size:train_size + val_size]

    test_flow = speed[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    train = DataSet(train_flow, train_labels)
    validation = DataSet(validation_speed, validation_labels)
    test = DataSet(test_flow, test_labels)
    Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])
    return Datasets(train=train, validation=validation, test=test)

train, validation, test = read_data_sets()
print(test.speed.shape)