import train_test as tt
import numpy as np 
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data = np.load('30_labelled.npy')

x_train, y_train, x_test, y_test = tt.get_train_test(data, 0.8)

clf = SVC()
param_name = 'gamma'
param_range = np.arange(0, 0.5, 0.025)
train_scores, test_scores = validation_curve(clf, x_train, y_train, 
  param_name=param_name, param_range=param_range, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, color='blue', label='train_scores')
plt.plot(param_range, test_scores_mean, color='orange', label='test_scores')

plt.title('Validation Curve, SVM')
plt.xlabel("$\gamma$")
plt.ylabel("Score (accuracy")

plt.legend(loc="best")

plt.savefig('gamma.png')