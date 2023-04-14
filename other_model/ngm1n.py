from egmn.gray import NGM1N
from matplotlib import pyplot as plt
import numpy as np
from optimization import GeneticAlgorithm


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

np.random.seed(42)

train_num = 24 # 16

def model(pars):
    x = pars[:, 0]
    loss = []
    for i in range(pars.shape[0]):
        reg = NGM1N(gamma=x[i])
        reg.fit(data[:train_num, :])
        reg.predict(data[train_num:, :])
        loss.append(reg.score())
    return np.asarray(loss)


ga = GeneticAlgorithm(func=model, dims=2, xlim=[[0.1, 1], [2, 2]], verbose=True, random_state=42)
ga.fit(display=True)
best_par = ga.best_

print(best_par)


reg = NGM1N(gamma=best_par[0])
reg.fit(data[:train_num, :])

y_hat0, prd = reg.predict(data[train_num:, :])

plt.plot(data[:, 0], 'r-')
plt.plot(y_hat0, 'g-o')
plt.show()


print(y_hat0)
for d in y_hat0:
    print(d)

