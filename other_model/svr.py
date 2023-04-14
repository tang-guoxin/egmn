from sklearn.svm import SVR
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


# data = np.loadtxt('../mydata/Australia.txt').T
# data = np.loadtxt('../mydata/Canada.txt').T
# data = np.loadtxt('../mydata/Finland.txt').T
# data = np.loadtxt('../mydata/Germany.txt').T
# data = np.loadtxt('../mydata/Hungary.txt').T
# data = np.loadtxt('../mydata/Poland.txt').T
# data = np.loadtxt('../mydata/UK.txt').T
# data = np.loadtxt('../mydata/Japan.txt').T
# data = np.loadtxt('../mydata/AustraliaPop.txt').T
# data = np.loadtxt('../mydata/UKPop.txt').T
# data = np.loadtxt('../mydata/LaosPop.txt').T
# data = np.loadtxt('../mydata/ThailandEnergy.txt').T
# data = np.loadtxt('../mydata/UAEEnergy.txt').T
# data = np.loadtxt('../mydata/AzerbaijanEnergy.txt').T
# data = np.loadtxt('../mydata/Power1.txt').T
# data = np.loadtxt('../mydata/Power2.txt').T
data = np.loadtxt('../mydata/Power3.txt').T

std_x, std_y = StandardScaler(), StandardScaler()

y_org = data[:, 0]

std_y.fit(data[:, 0].reshape(-1, 1))
std_x.fit(data[:, 1:])

x = std_x.transform(data[:, 1:])
y = std_y.transform(data[:, 0].reshape(-1, 1))

reg = SVR()

train_num = 24 # 16

reg.fit(x[:train_num, :], y[:train_num])
y_pred = reg.predict(x)

plt.plot(y, 'r-o')
plt.plot(y_pred, 'g-o')
plt.show()


y1 = std_y.inverse_transform(y_pred.reshape(-1, 1))

for d in y1:
    print(d[0])
