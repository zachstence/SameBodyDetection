import time
import numpy as np 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

def print_cm(cm):
  print("          actual")
  print("          F     T  ")
  print("pred  F   " + str(cm[0][0]) + "     " + str(cm[0][1]))
  print("      T   " + str(cm[1][0]) + "     " + str(cm[1][1]))


def get_train_test(data, ratio):
  shuffled = np.random.shuffle(data)

  i = int(ratio * len(data))
  train = data[:i]
  test = data[i:]

  x_train = list(train[:,0])
  y_train = list(train[:,1])

  x_test = list(test[:,0])
  y_test = list(test[:,1])

  return x_train, y_train, x_test, y_test



# class_weight = {True : , False : }

data = np.load('10_labelled.npy')


x_train, y_train, x_test, y_test = get_train_test(data, 0.8)

# train_true = np.sum(y_train)
# train_false = len(y_train) - train_true

# print(train_true)
# print(train_false)

balanced = {True : 9.30614525, False : 0.52838927}
custom = {True : 9.5, False : 0.5}

clf = SVC(class_weight=custom)
clf.fit(x_train, y_train)

print("trained")

y_pred = clf.predict(x_test)

print("tested")

cm = metrics.confusion_matrix(y_test, y_pred)
print_cm(cm)

# acc = metrics.accuracy_score(y_test, y_pred)
# print("accuracy: " + str(acc))

# f1 = metrics.f1_score(y_test, y_pred)
# print("f1: " + str(f1))

cr = metrics.classification_report(y_test, y_pred)
print(cr)