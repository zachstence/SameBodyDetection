import time
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import train_test as tt



p = 30 # want this to be high
w = 4  # want this to be low
cw = 9 # want this to be high
data = np.load('../data/' + 'p' + str(p) + '_w' + str(w) + '_cw' + str(cw) + '.npy')


x_train, y_train, x_test, y_test = tt.get_train_test(data, 0.8)


# RANDOM SEARCHING
'''
rf = RandomForestClassifier()

param_options = {
  'n_estimators' : range(100, 2001, 100),
  # 'criterion' : [],
  'max_features' : ['auto', 3, 4, 5],
  'max_depth' : [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  'min_samples_split' : [2, 5, 10],
  'min_samples_leaf' : [1, 2, 4],
  # 'min_weight_fraction_leaf' : [],
  # 'max_leaf_nodes' : [],
  # 'min_impurity_split' : [],
  # 'min_impurity_decrease' : [],
  'bootstrap' : [True, False]
  # 'oob_score' : [],
}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_options, 
  n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(x_train, y_train)

print(rf_random.best_params_)
'''

best = {'bootstrap': False, 'n_estimators': 600, 'min_samples_leaf': 1, 'max_depth': 40, 'max_features': 'auto', 'min_samples_split': 2}

rf = RandomForestClassifier(**best)

avg_acc = 0
avg_prec = 0
avg_f1 = 0
num = 20
for i in range(num):
  rf.fit(x_train, y_train)
  
  y_pred = rf.predict(x_test)
  
  avg_acc += accuracy_score(y_test, y_pred)
  avg_prec += precision_score(y_test, y_pred)
  avg_f1 += f1_score(y_test, y_pred)
  
avg_acc /= num
avg_prec /= num
avg_f1 /= num

print('Average accuracy:  ' + str(avg_acc))
print('Average precision: ' + str(avg_prec))
print('Average F1 Score:  ' + str(avg_f1))