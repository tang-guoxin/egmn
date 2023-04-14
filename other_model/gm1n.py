from egmn.gray import GM1N
import numpy as np
from matplotlib import pyplot as plt


# data = np.loadtxt('../mydata/Australia.txt').T
# data = np.loadtxt('../mydata/Canada.txt').T
# data = np.loadtxt('../mydata/Finland.txt').T
# data = np.loadtxt('../mydata/Germany.txt').T
# data = np.loadtxt('../mydata/Hungary.txt').T
# data = np.loadtxt('../mydata/Poland.txt').T
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


x, y = data[:, 1:], data[:, 0]

reg = GM1N()
train_num = 24 # 16
reg.fit(data[:train_num, :])

y_hat0, prd = reg.predict(data[train_num:, :])

plt.plot(data[:, 0], 'r-')
plt.plot(y_hat0, 'g-o')
plt.show()

print(y_hat0)
for d in y_hat0:
    print(d)
