import warnings
import tensorflow as tf
from matplotlib import pyplot as plt
from mydata.load import load_data
from egmn.core import EGMN
from optimization import GeneticAlgorithm
import numpy as np
from sklearn.metrics import mean_squared_error
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


data_path = [
    r'\mydata\Australia.txt',
    r'\mydata\Canada.txt',
    r'\mydata\Finland.txt',
    r'\mydata\Germany.txt',
    r'\mydata\Hungary.txt',
    r'\mydata\Poland.txt',
    r'\mydata\UK.txt',
    r'\mydata\Japan.txt',
    r'\mydata\AustraliaPop.txt',
    r'\mydata\UKPop.txt',
    r'\mydata\LaosPop.txt',
    r'\mydata\ThailandEnergy.txt',
    r'\mydata\UAEEnergy.txt',
    r'\mydata\AzerbaijanEnergy.txt',
    r'\mydata\Power1.txt',
    r'\mydata\Power2.txt',
    r'\mydata\Power3.txt',
]

np.random.seed(42)
tf.random.set_seed(42)

# train_num = 10
# test_num = 5

train_num = 20
test_num = 4


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
        # print(f'lambda = {lambda_g[i]}, cell_num = {cell_num[i]}, mse = {rmse}')
    return np.asarray(loss_list)


# ga = GeneticAlgorithm(func=model, dims=2, xlim=[[-1, 1], [6, 24]], population=100, verbose=True, max_iter=50)
# ga.fit(display=True)
# best_par = ga.best_
# print(best_par)

# best_par = [1.04622875e-02, 16]
# best_par = [-7.04495605e-02, 12]
# best_par = [-2.91224656e-02, 12]
# best_par = [3.83235951e-02, 14]
# best_par = [0.01940789, 12]
# best_par = [1.01018344e-01, 16]
# best_par = [0.01388076, 24]
# best_par = [0.01288076, 14]
# * appendix 2
# best_par = [0.0216, 15]
# best_par = [0.1240, 15]
# best_par = [0.030240, 14]

# best_par = [0.10700, 10]
# best_par = [0.05170, 14]
# best_par = [0.10212356, 12]

# best_par = [0.09981072, 12]
# best_par = [-0.0219981, 12]
best_par = [0.03723235, 12]

reg = EGMN(rt[:train_num], lr=0.01, max_iter=200, lambda_g=best_par[0], cell_num=int(best_par[1]), verbose=True)
reg.build(input_shape=[None, 1, x.shape[2]])
reg.summary()
reg.train(x[:train_num], y[1:train_num + 1])
y_fit = reg.call(inputs=x[:train_num])
y_pred = reg.predict_system(x, y[0])


y1 = std_y.inverse_transform(y_pred.reshape(-1, 1))

for d in y1:
    print(d[0])

plt.plot(y0, 'r-x')
plt.plot(y1, 'g-x')
plt.show()


