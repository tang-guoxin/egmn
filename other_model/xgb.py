import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.model_selection import HalvingGridSearchCV


# data = np.loadtxt('../mydata/Australia.txt').T
# data = np.loadtxt('../mydata/Canada.txt').T
# data = np.loadtxt('../mydata/Finland.txt').T
# data = np.loadtxt('../mydata/Germany.txt').T
# data = np.loadtxt('../mydata/Hungary.txt').T
# data = np.loadtxt('../mydata/Poland.txt').T
# data = np.loadtxt('../mydata/UK.txt').T
# data = np.loadtxt('../mydata/Japan.txt').T
# data = np.loadtxt('../mydata/AustraliaPop.txt').T
# data = np.loadtxt('../mydata/UKPop.txt').T # None
# data = np.loadtxt('../mydata/LaosPop.txt').T
# data = np.loadtxt('../mydata/ThailandEnergy.txt').T
# data = np.loadtxt('../mydata/UAEEnergy.txt').T
# data = np.loadtxt('../mydata/AzerbaijanEnergy.txt').T
# data = np.loadtxt('../mydata/Power1.txt').T
# data = np.loadtxt('../mydata/Power2.txt').T
data = np.loadtxt('../mydata/Power3.txt').T

x, y = data[:, 1:], data[:, 0]

parameters = {'n_estimators': [i for i in range(10, 500, 10)],
              'max_depth': [i for i in range(2, 10, 1)],
              'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
              'min_child_weight': [i for i in range(5, 21, 1)],
              'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
              }

reg = XGBRegressor()

hp = RandomizedSearchCV(estimator=reg, param_distributions=parameters, random_state=123, verbose=2)

# reg.fit(x[:16, :], y[:16])
# y_pred = reg.predict(x)

train_num = 24 # 16

hp.fit(x[:train_num, :], y[:train_num])
y_pred = hp.predict(x)

plt.plot(y, 'r-o')
plt.plot(y_pred, 'g-o')
plt.show()

for d in y_pred:
    print(d)



