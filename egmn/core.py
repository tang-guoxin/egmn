import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR as SVRModel
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


class EGMN(tf.keras.Model):
    def __init__(self, rt, lambda_g=0, cell_num=10, lr=0.01, max_iter=100, verbose=False, eps=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt = rt
        self.cell_num = cell_num
        self.lr = lr
        self.max_iter = max_iter
        self.lambda_g = lambda_g
        self.rnn = layers.SimpleRNN(self.cell_num, dropout=0.5)
        self.fc = layers.Dense(1)
        self.opt = optimizers.Adam(learning_rate=self.lr)
        self.verbose = verbose
        self.eps = eps

    def call(self, inputs, training=None, mask=None):
        y = self.rnn(inputs)
        z = self.fc(y)
        s = z - self.lambda_g * self.rt
        return s

    def elman(self, x):
        y = self.rnn(x)
        z = self.fc(y)
        return z

    def train(self, inputs, y_true):
        mse = losses.MeanSquaredError()
        for step in range(self.max_iter):
            with tf.GradientTape() as g:
                g.watch(self.trainable_weights)
                y = self.call(inputs)
                cost = mse(y_true=y_true, y_pred=y)
                if cost < self.eps:
                    print(f'step = {step + 1}, \t cost = {cost}.')
                    break
                if (step + 1) % 10 == 0 and self.verbose:
                    print(f'step = {step + 1}, \t cost = {cost}.')
                dw = g.gradient(cost, self.trainable_weights)
                self.opt.apply_gradients(zip(dw, self.trainable_weights))
        return cost

    def predict_system(self, x, y0):
        got = self.elman(x)
        row = x.shape[0]
        y_hat1 = [y0[0], ]
        for k in range(1, row + 1):
            ye = y0 * np.exp(-self.lambda_g * k)
            for tao in range(k):
                ep = k - tao - 1
                ye += np.exp(-self.lambda_g * (ep + 0.5)) * got[tao]
            y_hat1.append(ye.numpy()[0])
        y_hat1 = np.asarray(y_hat1)
        y_hat0 = np.hstack((y0, y_hat1[1:] - y_hat1[:-1]))
        return y_hat0


# * MLP
class MLP:
    def __init__(self, cell_num):
        self.cell_num = cell_num
        self.reg = MLPRegressor(hidden_layer_sizes=self.cell_num, activation='tanh', max_iter=100)

    def fit(self, x, y):
        self.reg.fit(x, y)
        return True

    def predict(self, x):
        y_pred = self.reg.predict(x)
        return y_pred


# * SVR
class SVR:
    def __init__(self):
        self.reg = SVRModel()
        pass

    def fit(self, x, y):
        self.reg.fit(x, y)
        return True

    def predict(self, x):
        y_pred = self.reg.predict(x)
        return y_pred


# * XGBoost
class XGB:
    def __init__(self):
        self.parameters = {'n_estimators': [i for i in range(10, 500, 10)],
                           'max_depth': [i for i in range(2, 10, 1)],
                           'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
                           'min_child_weight': [i for i in range(5, 21, 1)],
                           'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                           'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                           'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                           'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
                           }
        self.xgbr = RandomizedSearchCV(XGBRegressor(), self.parameters, cv=10, verbose=0, random_state=42)

    def fit(self, x, y):
        self.xgbr.fit(x, y)
        return True

    def predict(self, x):
        y_pred = self.xgbr.predict(x)
        return y_pred


class ElmanRNN(tf.keras.Model):
    def __init__(self, cell_num=10, lr=0.01, max_iter=100, verbose=False, eps=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_num = cell_num
        self.lr = lr
        self.max_iter = max_iter
        self.rnn = layers.SimpleRNN(self.cell_num, dropout=0.5)
        self.fc = layers.Dense(1)
        self.opt = optimizers.Adam(learning_rate=self.lr)
        self.verbose = verbose
        self.eps = eps

    def call(self, inputs, training=None, mask=None):
        y = self.rnn(inputs)
        z = self.fc(y)
        return z
