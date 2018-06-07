import time
import numpy as np 
from sklearn.svm import SVC

def get_train_test(data, ratio):
  shuffled = np.random.shuffle(data)

  i = int(ratio * len(data))
  train = data[:i]
  test = data[i:]

  x_train = train[:,0]
  y_train = train[:,1]

  x_test = test[:,0]
  y_test = test[:,1]

  return x_train, y_train, x_test, y_test




data = np.load('3_labelled.npy')


x_train, y_train, x_test, y_test = get_train_test(data, 0.7)

# print 'x_train', x_train
# print 'y_train', y_train
# print 'x_test', x_test
# print 'y_test', y_test

classifier = SVC()
classifier.fit(x_train, y_train)