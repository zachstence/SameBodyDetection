import numpy as np
from itertools import combinations
import feature_extraction as fe

def get_coherence_matrix(w, c, trial1, trial2):
  person1 = trial1[0]
  sig1 = trial1[1]
  person2 = trial2[0]
  sig2 = trial2[1]

  short = min(len(sig1), len(sig2))
  sig1 = sig1[:short]
  sig2 = sig2[:short]

  A = fe.get_feature_matrix(sig1, w)
  B = fe.get_feature_matrix(sig2, w)
  phi_max = 10
  return fe.N_matrix(A, B, c, phi_max)

def process(pairs, w, c):
  labelled_data = []
  for pair in pairs:
    print '(', pair[0][0], ',', pair[1][0], ')'
    labelled_data.append([get_coherence_matrix(w, c, *pair), (pair[0][0] == pair[1][0])])
  return labelled_data

# Loop through 2D data with indices [person #, trial #]
def get_all_trials(data):
  all_trials = []

  # shorten all streams to the length of the shortest
  shortest = len(data[0][0])
  for p in range(len(data)):
    for t in range(len(data[p])):
      mag = data[p][t]
      if len(mag) < shortest:
        shortest = len(mag)

  # append each magnitude stream to all_trials
  for p in range(len(data)):
    for t in range(len(data[p])):
      mag = data[p][t]
      mag = mag[:shortest]
      all_trials.append([p, mag])

  return all_trials

# This function returns all possible pairs of trials for "processing" later
def get_pairs(trials):
  return list(combinations(trials, 2))