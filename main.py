import tensorflow as tf
from matplotlib import pyplot as plt
from mydata.load import load_data
from egmn.core import EGMN
from optimization import GeneticAlgorithm
import numpy as np
from sklearn.metrics import mean_squared_error


data_path = [
    r'./mydata/Australia.txt',
    r'./mydata/Canada.txt',
    r'./mydata/Finland.txt',
    r'./mydata/Germany.txt',
    r'./mydata/Hungary.txt',
    r'./mydata/Poland.txt',
    r'./mydata/UK.txt',
    r'./mydata/Japan.txt',
]

np.random.seed(42)
tf.random.set_seed(42)

train_num = 10
test_num = 5

x, y, rt, std_y, y0 = load_data(data_path[-1])


def model(par):
    lambda_g, cell_num = par[:, 0], np.floor(par[:, 1])
    loss_list = []
    for i in range(par.shape[0]):
        reg = EGMN(rt[:train_num], lr=0.01, max_iter=100, lambda_g=lambda_g[i], cell_num=int(cell_num[i]))
        reg.build(input_shape=[None, 1, x.shape[2]])
        reg.train(x[:train_num], y[1:train_num + 1])
        y_pred = reg.predict_system(x, y[0])
        rmse = mean_squared_error(y_true=y[train_num + test_num + 1:], y_pred=y_pred[train_num + test_num + 1:]) ** 0.5
        loss_list.append(rmse)
    return np.asarray(loss_list)


ga = GeneticAlgorithm(func=model, dims=2, xlim=[[-2, 2], [6, 16]], population=100, verbose=True, max_iter=50)
ga.fit(display=True)
best_par = ga.best_
print(best_par)

reg = EGMN(rt[:train_num], lr=0.01, max_iter=200, lambda_g=best_par[0], cell_num=int(best_par[1]), verbose=True)
reg.build(input_shape=[None, 1, x.shape[2]])
reg.summary()
reg.train(x[:train_num], y[1:train_num + 1])
y_fit = reg.call(inputs=x[:train_num])
y_pred = reg.predict_system(x, y[0])
y1 = std_y.inverse_transform(y_pred.reshape(-1, 1))


plt.plot(y0, 'r-x')
plt.plot(y1, 'g-x')
plt.show()

