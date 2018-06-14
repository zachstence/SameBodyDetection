import scipy.io as sio
import numpy as np
import choose

# Import MatLab data as a dictionary
mat = sio.loadmat('./UniMiB-SHAR/data/full_data.mat')

# Get numpy arrays containing data
data = mat['full_data']

num_people = 15
data = data[:num_people]

new_data = []

# Loop through dataset and append trials to new_data
for person_index in range(len(data)):
  accel_data = data[person_index][0][0][0]
  activity_index = 2
  activity = accel_data[activity_index]

  # Loop through trials and append magnitude streams to trial list
  trials = []
  for t in range(len(activity)):
    trial = activity[t][0]
    magnitude = trial[5]
    trials.append(magnitude)

  new_data.append(trials)


trials = choose.get_all_trials(new_data)

pairs = choose.get_pairs(trials)

# Samples per window
w = 4
# Seconds per coherence calculation
coherence_window = 2

c = choose.fe.get_c(w, coherence_window)

labelled = choose.process(pairs, w, c)

rows = choose.get_all_rows(labelled)

filename = ('p' + str(num_people) + '_w' + str(w) + '_cw' + str(coherence_window)) + '.npy'
np.save(filename, rows)