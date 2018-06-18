import time
import numpy as np 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd


def print_cm(cm):
  print("           predicted")
  print("            F     T  ")
  print("actual  F   " + str(cm[0][0]) + "     " + str(cm[0][1]))
  print("        T   " + str(cm[1][0]) + "     " + str(cm[1][1]))



####################################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor

dict_classifiers = {
    # "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    # "Neural Net": MLPClassifier(alpha = 1),
    # "Naive Bayes": GaussianNB(),
    # "AdaBoost": AdaBoostClassifier(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    # "Gaussian Process": GaussianProcessRegressor()
}


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(Y_test, y_pred)
        print_cm(cm)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models
 

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))

############################################################################################






















def get_true_false_count(y):
  true_count = 0
  false_count = 0
  for b in y:
    if b:
      true_count += 1
    else:
      false_count += 1
  return true_count, false_count


def balance_data(data):
  true_data = []
  false_data = []
  for row in data:
    row = list(row)
    if row[1] == True:
      true_data.append(row)
    else:
      false_data.append(row)
  true_length = len(true_data)

  np.random.shuffle(false_data)
  false_data = false_data[:true_length]

  all_data = true_data + false_data
  return all_data

def get_train_test(data, ratio):
  data = balance_data(data)
  np.random.shuffle(data)

  i = int(ratio * len(data))
  train = data[:i]
  test = data[i:]

  x_train = [ row[0] for row in train ]
  y_train = [ row[1] for row in train ]

  x_test = [ row[0] for row in test ]
  y_test = [ row[1] for row in test ]

  return x_train, y_train, x_test, y_test





p = 30 # want this to be high
w = 4  # want this to be low
cw = 9 # want this to be high
data = np.load('./data/' + 'p' + str(p) + '_w' + str(w) + '_cw' + str(cw) + '.npy')


x_train, y_train, x_test, y_test = get_train_test(data, 0.8)

dict_models = batch_classify(x_train, y_train, x_test, y_test)
display_dict_models(dict_models)