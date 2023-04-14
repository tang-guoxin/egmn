from abc import ABC

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import api as sm

"""
第一列为需要fit的值
"""

""":key
GM(1, n) model
"""


class GM1N:
    def __init__(self):
        self.is_fit = False
        self.y0 = None
        self.y_hat0 = None
        self.par = None
        self.y1 = None
        self.pred = None

    def one_ago(self, y):
        return np.cumsum(y, axis=0)

    def make_mat_B(self, y, y01):
        y1 = y[:, 0]
        z1 = -0.5 * (y1[1:] + y1[:-1])
        B = np.concatenate((z1.reshape(-1, 1), y[1:, 1:]), axis=1)
        yn = y01[1:].reshape(-1, 1)
        par = np.linalg.inv(B.T @ B) @ B.T @ yn
        self.par = par[:, 0]
        return self.par

    def fit(self, y0):
        self.yy = y0
        self.y0 = y0[:, 0]
        self.y1 = self.one_ago(y0)
        par = self.make_mat_B(self.y1, y0[:, 0])
        self.is_fit = True
        return None

    def predict(self, xp):
        # xp = self.one_ago(xp)
        y1p = np.concatenate((self.yy, xp), axis=0)
        y1p = self.one_ago(y1p)
        n, dim = y1p.shape[0], self.par.shape[0]
        train_num = self.y1.shape[0]
        y_hat1, y_hat0 = np.zeros(n), np.zeros(n)
        y_hat1[0], y_hat0[0] = self.y0[0], self.y0[0]
        a = self.par[0]
        for k in range(1, n):
            add1 = 0
            for i in range(1, dim):
                add1 += (self.par[i] * y1p[k, i] / a)
            y_hat1[k] = (self.y0[0] - add1) * np.exp(-a * k) + add1
        for i in range(n - 1, 0, -1):
            y_hat0[i] = y_hat1[i] - y_hat1[i - 1]
        self.pred = y_hat0
        prd = y_hat0[train_num:]
        return y_hat0, prd

    def score(self, xp):
        n = self.y1.shape[0]
        y_true = xp[:, 0]
        y_pred = self.pred[n:]
        print(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print('r2 = %.4f' % r2)
        print('mse = %.4f' % mse)
        return r2, mse


class NGM1N:
    def __init__(self, gamma):
        self.gamma = gamma
        self.is_fit = False
        self.y0 = None
        self.y_hat0 = None
        self.par = None
        self.y1 = None
        self.pred = None
        self.yy = None

    def fit(self, y0):
        self.yy = y0
        self.y0 = y0[:, 0]
        self.y1 = self.one_ago(y0)
        par = self.make_mat_B(self.y1, y0[:, 0])
        self.is_fit = True
        return None

    def one_ago(self, y):
        return np.cumsum(y, axis=0)

    def pow(self, A):
        ret = np.zeros_like(A)
        [m, n] = A.shape
        for i in range(m):
            for j in range(n):
                ret[i, j] = (A[i, j]) ** self.gamma
        return ret

    def make_mat_B(self, y, y01):
        y1 = y[:, 0]
        z1 = -0.5 * (y1[1:] + y1[:-1])
        B = np.concatenate((z1.reshape(-1, 1), self.pow(y[1:, 1:])), axis=1)
        # print(y[1:, 1:])
        # print(np.power(y[1:, 1:], self.gamma))
        yn = y01[1:].reshape(-1, 1)
        par = np.linalg.inv(B.T @ B) @ B.T @ yn
        self.par = par[:, 0]
        return self.par

    def predict(self, xp):
        y1p = np.concatenate((self.yy, xp), axis=0)
        y1p = self.one_ago(y1p)
        n, dim = y1p.shape[0], self.par.shape[0]
        train_num = self.y1.shape[0]
        y_hat1, y_hat0 = np.zeros(n), np.zeros(n)
        y_hat1[0], y_hat0[0] = self.y0[0], self.y0[0]
        a = self.par[0]
        for k in range(1, n):
            add1 = 0
            for i in range(1, dim):
                tmp = y1p[k, i]
                add1 += (self.par[i] * (tmp ** self.gamma) / a)
            y_hat1[k] = (self.y0[0] - add1) * np.exp(-a * k) + add1
        for i in range(n - 1, 0, -1):
            y_hat0[i] = y_hat1[i] - y_hat1[i - 1]
        self.pred = y_hat0
        self.y_hat0 = y_hat0
        prd = y_hat0[train_num:]
        return y_hat0, prd

    def score(self, sco="fit"):
        mse = None
        if sco == "fit":
            mse = mean_squared_error(self.y0, self.y_hat0[:self.y0.shape[0]])
            # print('mse = %.4f' % mse)
        elif sco == 'test':
            pass
        else:
            pass
        return mse


class DGM1N:
    def __init__(self):
        self.is_fit = False
        self.y_hat0 = None
        self.par = None
        self.pred = None
        self.y = None
        self.x1 = None
        self.y1 = None

    def fit(self, y):
        self.y = y
        y = np.cumsum(y, axis=0)
        y1 = y[:, 0]
        x1 = y[:, 1:]
        self.y1 = y1
        self.x1 = y1
        n = y.shape[0]
        B = np.concatenate((y1[:-1].reshape(-1, 1), x1[1:, :], np.ones((n - 1, 1))), axis=1)
        par = np.linalg.inv(B.T @ B) @ B.T @ y1[1:]
        self.par = par

    def predict(self, test):
        y1p = np.concatenate((self.y, test), axis=0)
        y1p = np.cumsum(y1p, axis=0)
        # print(y1p.shape)
        y1 = (y1p[:, 0])
        x1 = (y1p[:, 1:])
        # print(x1.shape)
        n = y1p.shape[0]
        ss = np.zeros(n)
        ss[0] = self.y[0, 0]
        one = np.ones((y1.shape[0] - 1, 1))
        ss[1:] = np.concatenate((y1[1:].reshape(-1, 1), x1[:-1, :], one), axis=1) @ self.par
        sim = np.zeros_like(ss)
        sim[0] = self.y[0, 0]
        sim[1:] = ss[1:] - ss[:-1]
        y_hat0 = sim
        self.pred = y_hat0
        y_pred = sim[self.y.shape[0]:]
        return y_hat0, y_pred


class GMC1N:
    def __init__(self, rp) -> None:
        self.is_fit = False
        self.rp = rp
        self.y_hat0 = None
        self.pars = None
        self.x = None
        self.y_hat0 = None
        self.y0 = None

    def fit(self, x) -> None:
        self.x = x
        xago, n = np.cumsum(x, axis=0), x.shape[0]
        y1, x1, y0 = xago[:, 0], xago[:, :-1], x[:, 0]
        self.y0 = y0
        z1 = -0.5 * (y1[1:] + y1[:-1])
        zi = 0.5 * (x1[1:] + x1[:-1])
        yr = y0[1:].reshape(-1, 1)
        mat_b = np.concatenate((z1.reshape(-1, 1), zi, np.ones((n - 1, 1))), axis=1)
        # print(mat_b.shape, yr.shape)
        pars = np.linalg.inv(mat_b.T @ mat_b) @ mat_b.T @ yr
        self.pars = pars[:, 0]
        self.is_fit = True

    def ft(self, x1, t):
        u = self.pars[-1]
        s = self.pars[:-1] @ x1[t, :].reshape(-1, 1) + u
        return s

    def predict(self, test_x):
        x = np.concatenate((self.x, test_x), axis=0)
        x1 = np.cumsum(x, axis=0)
        u = lambda v: 0 if v < 4 else 1
        n = x.shape[0]
        y_hat1 = x1[0, :]
        x1 = x1[:, 1:]
        for t in range(1, n - self.rp):
            constx = self.y0[self.rp] * np.exp(-self.pars[0] * (t - 1))
            cs = 0
            for tao in range(1, t):
                cs += np.exp(-self.pars[0] * (t - tao - 0.5)) * (0.5 * (self.ft(x1, tao) + self.ft(x1, tao - 1)))
            cs *= u(t)
            y_hat1[self.rp + t] = cs + constx
        self.y_hat0 = y_hat1[1:] - y_hat1[:-1]
        return self.y_hat0


class GMCO(ABC):
    def __init__(self):
        super(GMCO, self).__init__()
        pass

    def get_params(self, x, y):
        n = y.shape[0]
        y = np.reshape(y, (-1, 1))
        y1 = np.cumsum(y, axis=0)
        x1 = np.cumsum(x, axis=0)
        zy1 = 0.5 * (y1[1:] + y1[:-1])
        zx1 = 0.5 * (x1[1:, :] + x1[:-1, :])
        mat = np.hstack((zy1, zx1, np.ones(shape=(n - 1, 1))))
        reg = sm.OLS(y[1:, :], mat)
        res = reg.fit()
        pars = res.params
        return pars, mat

    def fit(self, x, y, n):
        y = np.reshape(y, (-1, 1))
        [p, _] = self.get_params(x[:n, :], y[:n, :])
        len_ = y.shape[0]
        x1 = np.cumsum(x, axis=0)
        a = -p[0]
        bi = np.reshape(p[1:-1], (-1, 1))
        b = p[-1]
        # opt pars
        m = np.log((1 + 0.5 * a) / (1 - 0.5 * a))
        bi = bi / np.sqrt((1 - 0.25 * (a ** 2)))
        b = b / np.sqrt((1 - 0.25 * (a ** 2)))
        a = m
        # params = np.hstack([a, bi[:, 0], b])
        # do
        sim1 = [y[0, 0], ]
        f = x1 @ bi + b
        f = f[:, 0]
        ct = list()
        for t in range(1, len_):
            tempc = 0
            for k in range(1, t + 1):
                tempc = tempc + 0.5 * (np.exp(-a * (t - k + 0.5)) * (f[k - 1] + f[k]))
            ct.append(tempc)
        for k in range(1, len_):
            yk = y[0, 0] * np.exp(-a * k) + ct[k - 1]
            sim1.append(yk)
        y_hat_1 = np.asarray(sim1)
        res = y_hat_1[1:] - y_hat_1[:-1]
        y_hat_0 = np.hstack([y[0, 0], res])
        return y_hat_0

# data = np.loadtxt('E:\论文\gray_memory_system\my_code\data\case1\data.txt')
# GN = GMC1N(rp=0)
#
# data = np.fliplr(data)
#
# train, test = data[:36, :], data[36:, :]
#
# GN.fit(train)
# GN.predict(test)
#
# print(GN.pars)

# class FGM1N:
#     def __init__(self, r):
#         self.r = r
#         pass


# if __name__=='__main__':
#     data = np.loadtxt('./data/data.txt')
#
#     data = np.fliplr(data)
#     train = data[:24, :]
#     test = data[24:, :]
#
#     GM = GM1N()
#
#     GM.fit(train)
#
#     y_hat0, pred_test = GM.predict(test)
#
#     prd = GM.pred
#
#     print(GM.par)
#
#     GM.score(test)
#
#     plt.plot(prd, 'g--x')
#     plt.plot(data[:, 0], 'r--o')
#     plt.legend(['prd', 'org'])
#     plt.show()
#
