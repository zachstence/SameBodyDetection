import scipy.io as sio
import numpy as np
import choose
from tempfile import TemporaryFile

def pprint(arr):
  return "[ " + str(arr[0]) + " ... ]"
np.set_string_function(pprint)





'''
# Import MatLab data as a dictionary
mat = sio.loadmat('./UniMiB-SHAR/data/full_data.mat')

# Get numpy arrays containing data
data = mat['full_data']

short = 3
data = data[:short]
print len(data)

new_data = []

# Loop through data and extract all fields into variables
for p in range(len(data)):
  accel_data = data[p][0][0][0]
  a = 2
  activity = accel_data[a]
  trials = []

  for t in range(len(activity)):
    trial = activity[t][0]
    magnitude        = trial[5]
    trials.append(magnitude)

  new_data.append(trials)


trials = choose.get_all_trials(new_data)
# for trial in trials:
#   print trial

pairs = choose.get_pairs(trials)
# for pair in pairs:
#   print type(pair[0]), type(pair[1])


DT = 0.02
w = 5
coherence_window = 3
c = int((coherence_window) / (w * DT))
labelled = choose.process(pairs, w, c)

np.save('out.npy', labelled)
'''

loaded = np.load('out.npy')
print loaded

print len(loaded)
