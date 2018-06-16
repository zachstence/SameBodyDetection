import scipy.io as sio
import numpy as np
import choose

mat = sio.loadmat('./UniMiB-SHAR/data/full_data.mat')
full_data = mat['full_data']

# people = [30]
# windows = [4, 8]
# coherence_windows = [5, 7, 9]

people = [15]
windows = [4]
coherence_windows = [9]

for p in people:
  data = full_data[:p]

  new_data = []
  # Loop through dataset and append trials to new_data
  for person_index in range(len(data)):
    accel_data = data[person_index][0][0][0]
    activity_index = 2 # walking
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

  for w in windows:
    for cw in coherence_windows:
      filename = 'p' + str(p) + '_w' + str(w) + '_cw' + str(cw) + '.npy'
      try:
        print(filename)
        
        c = choose.fe.get_c(w, cw)

        labelled = choose.process(pairs, w, c)

        rows = choose.get_all_rows(labelled)

        np.save('./data/' + filename, rows)
      except KeyboardInterrupt:
        raise
      except:
        print('- ' + filename + ' skipped')
        continue
      finally:
        print('- ' + filename + ' finished')