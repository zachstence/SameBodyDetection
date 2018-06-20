import time
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import train_test as tt



p = 30 # want this to be high
w = 4  # want this to be low
cw = 9 # want this to be high
data = np.load('../data/' + 'p' + str(p) + '_w' + str(w) + '_cw' + str(cw) + '.npy')


x_train, y_train, x_test, y_test = tt.get_train_test(data, 0.8)


# clf = KNeighborsClassifier()
clf = RandomForestClassifier(n_estimators=100, max_features=4, min_samples_leaf=1)


param_name = 'min_samples_leaf'
param_range = range(1, 20)
scoring = 'accuracy'
train_scores, test_scores = validation_curve(clf, x_train+x_test, y_train+y_test, 
  param_name=param_name, param_range=param_range, cv=10, scoring=scoring)

# print(np.shape(train_scores))
# print(np.shape(test_scores))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# print(param_range)
# print(train_scores_mean)
# print(test_scores_mean)

plt.plot(param_range, train_scores_mean, color='b')
plt.plot(param_range, test_scores_mean, color='g')

plt.xlabel(param_name)
plt.ylabel(scoring)


plt.savefig(param_name + '_' + scoring + '.png')

