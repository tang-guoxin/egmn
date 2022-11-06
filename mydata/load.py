import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = np.loadtxt(path).T
    std_x, std_y = StandardScaler(), StandardScaler()
    x, y = data[:, 1:], data[:, 0]
    y0 = y
    x = std_x.fit_transform(x)
    x = np.cumsum(x, axis=1)
    x = np.hstack((x[:-1, :], x[1:, :]))
    x = tf.expand_dims(x, axis=1)
    std_y.fit(y.reshape(-1, 1))
    y = std_y.transform(y.reshape(-1, 1))
    y1 = np.cumsum(y, axis=1)
    rt = 0.5 * (y1[1:] + y1[:-1])
    return x, y, rt, std_y, y0

