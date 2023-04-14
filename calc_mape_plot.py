import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# ! index
# For section 4, n = 15
n = 24
data = np.loadtxt('./res.txt') # / 10000
train, test = data[:, :n], data[:, n:]
y_train, y_test = data[0, :n], data[0, n:]

# ! calc
train_metrics, test_metrics = np.zeros((3, 7)), np.zeros((3, 7))
for i in range(7):
    train_metrics[0, i] = mean_absolute_percentage_error(y_train, train[i+1, :])
    train_metrics[1, i] = np.sqrt(mean_squared_error(y_train, train[i+1, :]))
    train_metrics[2, i] = mean_absolute_error(y_train, train[i+1, :])
    # * test
    test_metrics[0, i] = mean_absolute_percentage_error(y_test, test[i+1, :])
    test_metrics[1, i] = np.sqrt(mean_squared_error(y_test, test[i+1, :]))
    test_metrics[2, i] = mean_absolute_error(y_test, test[i+1, :])


print(train_metrics)

print(test_metrics)

np.savetxt('train-metrics.txt', train_metrics, delimiter='\t', fmt='%.4f')
np.savetxt('test-metrics.txt', test_metrics, delimiter='\t', fmt='%.4f')

print(np.argmin(train_metrics, axis=1))
print(np.argmin(test_metrics, axis=1))
