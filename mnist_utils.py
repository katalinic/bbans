import numpy as np
import tensorflow.keras.datasets.mnist as mnist


def load_train():
    (train_data, _), _ = mnist.load_data()
    return train_data


def load_test():
    _, (test_data, _) = mnist.load_data()
    return test_data


def binarise(data):
    data = data.reshape(-1, 784)
    data = data.astype('float32') / 255.0
    data = (data > 0.5).astype(np.int)
    return data


def generate_batches(data, batch_size, seed=0):
    N = data.shape[0]
    inds = np.arange(N)
    np.random.RandomState(seed).shuffle(inds)
    data = data[inds]
    for i in range(0, N, batch_size):
        yield data[i:i + batch_size]
