from matplotlib import pyplot as plt
import numpy as np

class ParticleSwarmOptimization:
    def __init__(self,
                 func,
                 dims,
                 *,
                 xlim=(),
                 vlim=(),
                 pap=20,
                 w=(0.09, 0.99),
                 c1=1.65,
                 c2=1.65,
                 max_iter=100,
                 slow=None,
                 tol=1e-6
                 ):
        self.func = func
        self.dims = dims
        self.pap = pap
        self.xlim = xlim
        self.vlim = vlim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.best_ = None
        self.minf_ = np.inf
        self.x = np.random.uniform(xlim[0], xlim[1], (pap, dims))
        self.v = np.random.uniform(vlim[0], vlim[1], (pap, dims))
        self.his = []
        if slow is None:
            slow = max_iter
        self.slow = slow

    def condition(self, x, limit):
        c = np.zeros_like(x)
        for d in range(self.dims):
            tmp = x[:, d]
            tmp[tmp < limit[0][d]] = limit[0][d]
            tmp[tmp > limit[1][d]] = limit[1][d]
            c[:, d] = tmp
        return c

    def refresh_v(self, leader, follow, iter) -> bool:
        self.v = self.v + self.c1 * np.random.rand() * (leader - self.x) + self.c2 * np.random.rand() * (
                    follow - self.x)
        dert = self.w[0] + (self.w[1] - self.w[0]) * (self.max_iter - iter) / self.max_iter
        self.v = dert * self.v
        self.v = self.condition(x=self.v, limit=self.vlim)
        return True

    def refresh_follow(self, y, follow):
        for i in range(self.pap):
            if self.func(self.x[i, :].reshape(1, -1)) < y[i]:
                follow[i, :] = self.x[i, :]
        return follow

    def curve(self):
        plt.plot(self.his, 'r--.')
        plt.show()
        return True

    def fit(self, display=False, curve=False):
        times = 0
        follow = self.x
        y = self.func(follow)
        idx = np.argmin(y)
        leader = follow[idx, :]
        self.his.append(y[idx])
        for i in range(self.max_iter):
            self.refresh_v(leader=leader, follow=follow, iter=i)
            self.x = self.v + self.x
            self.x = self.condition(x=self.x, limit=self.xlim)
            follow = self.refresh_follow(y, follow=follow)
            y = self.func(follow)
            idx = np.argmin(y)
            leader = follow[idx, :]
            self.his.append(y[idx])
            if display:
                print('iter = %d. min = %f' % (i + 1, y[idx]))
            if np.abs(self.his[-1] - self.his[-2]) <= self.tol:
                times += 1
            if np.abs(self.his[-1] - self.his[-2]) > self.tol:
                times = 0
            if times > self.slow:
                break
        self.best_ = leader
        self.minf_ = np.min(y)
        if curve:
            self.curve()
        return np.min(y)



