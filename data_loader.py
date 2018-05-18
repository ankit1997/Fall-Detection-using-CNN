"""
@author: Ankit Bindal
"""

import os
import config
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

TIMESTEPS, NUM_FEATURES = config.FEATURES_SHAPE

class Data:
    def __init__(self, path):
        assert '_' not in path, "Please rename {} to replace underscores '_'".format(path)

        self.X = []
        self.Y = []
        subjects = self._getDirs(path)

        fallDirs = []
        nonFallDirs = []
        for subject in subjects:
            dirs = self._getDirs(subject)
            for d in dirs:
                if "ADL" in d:
                    nonFallDirs.append(d)
                elif "FALLS" in d:
                    fallDirs.append(d)
                else:
                    print("Unreachable code.")

        fallFiles = self._get_class_files(fallDirs)
        nonFallFiles = self._get_class_files(nonFallDirs)
        files = fallFiles + nonFallFiles
        files.sort()

        prefixes = list(set([x.split('_')[0] for x in files]))
        batches = []
        for pref in prefixes:
            batches.append([x for x in files if x.startswith(pref)])

        for i in tqdm(range(len(batches))):
            batch = batches[i]
            n = len(batch)

            assert n%3 == 0
            trails = n//3
            class_ = 0 if "ADL" in batch[0] else 1

            for i in range(trails):
                acc = batch[i]
                gyro = batch[i+trails]
                ori = batch[i+2*trails]

                data = self._read_trail(acc, gyro, ori)
                self.X.append(data)

                if class_ == 0:
                    label = [1, 0]
                elif class_ == 1:
                    label = [0, 1]
                else:
                    label = None
                    print("ERROR: Unknown class number: {}".format(class_))

                for i in range(len(data)):
                    self.Y.append(label)
        
        self.X = np.concatenate(self.X)
        self.Y = np.array(self.Y)

        print("X: {}".format(self.X.shape))
        print("Y: {}".format(self.Y.shape))

        # perform data augmentation
        self._augment()

    def _read_trail(self, acc, gyro, ori):
        '''
            Reads a trail given acc, gyro and ori sensor info files.
        '''
        acc_data = self._read_file(acc)
        gyro_data = self._read_file(gyro)
        # ori_data = self._read_file(ori)

        acc_data = self._adjust_len(acc_data)
        gyro_data = self._adjust_len(gyro_data)
        # ori_data = self._adjust_len(ori_data)

        combine = []
        for i in range(len(acc_data)):
            # combine.append(acc_data[i] + gyro_data[i] + ori_data[i])
            combine.append(acc_data[i] + gyro_data[i])
        combine = np.array(combine)
        
        if 0:
            sqrs = np.square(combine)
            sos = np.sum(sqrs, axis=1)
            sqrt = np.sqrt(sos)

            return np.expand_dims(sqrt, 0)
        else:
            return np.expand_dims(combine, 0)

    def _read_file(self, fname):
        data = open(fname).read().strip().split('\n')
        try:
            start = data.index('@DATA')+1
        except ValueError:
            print("Improper header information in {}".format(fname))
            exit()

        features = []
        data = data[start: ]
        for line in data:
            cols = line.split(',')
            cols = list(map(float, cols))
            features.append(cols[1:]) # remove timestamp information

        return features

    def _adjust_len(self, data):
        n = len(data)
        if n >= TIMESTEPS:
            return data[:TIMESTEPS]
        else:
            data = data + [[0.0] * 3] * (TIMESTEPS-n)
            return data
        
    def _get_class_files(self, directories):
        files = []
        for directory in directories:
            dirs = self._getDirs(directory)
            for d in dirs:
                files.extend(self._getFiles(d))

        assert all([f.endswith('.txt') for f in files]), "Unexpected file extension found"
        return files

    def _getDirs(self, path):
        return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def _getFiles(self, path):
        return [os.path.join(path, d) for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]

    def _augment(self):
        if np.random.random() > 0.5:
            x = self.X.copy()
            y = self.Y.copy()

            # Add random [0, 2) noise
            x = x + 2*np.random.random(size=x.shape)
            self.X = np.concatenate([self.X, x], axis=0)
            self.Y = np.concatenate([self.Y, y], axis=0)

        ##################################################

        if np.random.random() > 0.5:
            x = self.X.copy()
            y = self.Y.copy()

            # Subtract random [0, 2) noise
            x = x - 2*np.random.random(size=x.shape)
            self.X = np.concatenate([self.X, x], axis=0)
            self.Y = np.concatenate([self.Y, y], axis=0)

        ##################################################

        if np.random.random() > 0.5:
            x = self.X.copy()
            y = self.Y.copy()

            x = x + np.random.normal(size=self.X.shape)
            self.X = np.concatenate([self.X, x], axis=0)
            self.Y = np.concatenate([self.Y, y], axis=0)

class DataLoader:
    def __init__(self, path, use_previous=True, train_size=0.8):

        self.data = self.get_data(path, use_previous)

        self.X = self.data.X
        self.Y = self.data.Y

        # Shuffle dataset
        indices = list(range(self.X.shape[0]))
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)

        self.X = self.X[indices]
        self.Y = self.Y[indices]

        # Degree of balance in data
        num_fall = np.count_nonzero(np.argmax(self.Y, axis=1) == 1)
        num_nonFall = np.count_nonzero(np.argmax(self.Y, axis=1) == 0)
        print("(Fall, Non-fall) count = ({}, {})".format(num_fall, num_nonFall))

        self.trainX, self.validX, self.trainY, self.validY = train_test_split(self.X, self.Y, 
                                                                            train_size=train_size)
        print("Train dataset: {} , {}".format(self.trainX.shape, self.trainY.shape))
        print("Validation dataset: {} , {}".format(self.validX.shape, self.validY.shape))

        # print("Total training size: {}".format(self.trainX.shape[0] * self.trainX.shape[1] * self.trainX.shape[2]))

        num_points = self.trainX.shape[0]
        self.train_batches = int(np.ceil(num_points / config.BATCH_SIZE))

        num_points = self.validX.shape[0]
        self.valid_batches = int(np.ceil(num_points / config.BATCH_SIZE))

    def next_train(self):
        num_points = self.trainX.shape[0]
        for i in range(self.train_batches):
            start = i
            end = min(start+config.BATCH_SIZE, num_points)
            yield self.trainX[start: end], self.trainY[start: end]

    def next_validation(self):
        num_points = self.validX.shape[0]
        for i in range(self.valid_batches):
            start = i
            end = min(start+config.BATCH_SIZE, num_points)
            yield self.validX[start: end], self.validY[start: end]

    def get_data(self, path, use_previous):
        path_name = os.path.join(config.databin, "dataset.pkl")
        
        if use_previous:
            if os.path.isfile(path_name):
                prev_data = pickle.load(open(path_name, 'rb'))
                print("Using previous data...")
                return prev_data
        
        print("Loading data...")
        data = Data(path)
        pickle.dump(data, open(path_name, 'wb'))
        return data

if __name__ == "__main__":
    import sys
    d = DataLoader(sys.argv[1])