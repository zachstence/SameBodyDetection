import numpy as np
from itertools import combinations

def get_coherence_matrix(tuple1, tuple2):
  return str(tuple1) + str(tuple2)

def process(pairs):
  labelled_data = []
  for pair in pairs:
    tuple1 = pair[0]
    tuple2 = pair[1]
    labelled_data.append([get_coherence_matrix(tuple1, tuple2), (pair[0][0] == pair[1][0])])
  return labelled_data

# This function loops through a 2D array of data and ...
def get_all_trials(data):
  all_trials = []
  for p in range(len(data)):
    for t in range(len(data[p])):
      # with actual data we will be appending either a magnitude signal or feature matrix
      all_trials.append([p, data[p][t]])
  return all_trials

# This function returns all possible pairs of trials for "processing" later
def get_pairs(trials):
  return list(combinations(trials, 2))





data = []
person = [0, 1]
for i in range(30):
  data.append(person)

trials = get_all_trials(data)
# print 'all_trials:', trials
pairs = get_pairs(trials)
# print 'combinations:', pairs

output = process(pairs)
for o in output:
  print o