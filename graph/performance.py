import train_test as tt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_average(classifier, runs, x_train, y_train, x_test, y_test):
  average = 0
  for r in range(runs):
    classifier.fit(x_train, y_train)
    average += classifier.score(x_test, y_test)

  return average / runs
  

p = 30
w = 4
cws = [5, 7, 9]

stats = dict.fromkeys(tt.dict_classifiers.keys(), [])

for name in tt.dict_classifiers.keys():
  print(name)
  averages = []
  for cw in cws:
    print(cw)
    data = np.load('./data/' + 'p' + str(p) + '_w' + str(w) + '_cw' + str(cw) + '.npy')
    x_train, y_train, x_test, y_test = tt.get_train_test(data, 0.8)
    
    runs = 10
    average = get_average(tt.dict_classifiers[name], runs, x_train, y_train, x_test, y_test)
    averages.append(average)
  
  stats[name] = averages

print(stats)

colors = ['y', 'g', 'r', 'c', 'm']
plt.style.use('dark_background')

for name, color in zip(stats.keys(), colors):
  plt.plot(cws, stats[name], color, label=name)

plt.xlabel('Coherence Window (seconds)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Coherence Window of Several Classifiers')
plt.xticks([5, 7, 9])
plt.yticks(np.arange(0.5, 1, 0.1))
plt.legend()


plt.savefig('avg10.png', dpi=600, transparent=True)
