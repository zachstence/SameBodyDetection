import scipy.io as scipy
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

####################################### DATASET INFO ###################################################
# each row contains data of one subject -> 30 rows
  # each subject has accel data, gender, age, height, weight -> 5 fields
    # each field is a 2D numpy array containing the specified information
      # accel data  - an array of 17 activities (one for each recorded activity) each in a numpy array
        # each activity consists of either 2 or 6 trials 
          # each trial consists of 6 rows
            # x-axis acceleration
            # y-axis acceleration
            # z-axis acceleration
            # time in nanoseconds
            # time in seconds
            # magnitude of acceleration
      # gender      - a character (M/F)
      # age         - an unsiged 8-bit integer
      # height      - an unsiged 8-bit integer
      # weight      - an unsiged 8-bit integer
########################################################################################################

# List of activities in dataset
ACTIVITIES = ['StandingUpFS',
              'StandingUpFL',
              'Walking',
              'Running',
              'GoingUpS',
              'Jumping',
              'GoingDownS',
              'LyingDownFS',
              'SittingDown',
              'FallingForw',
              'FallingRight',
              'FallingBack',
              'HittingObstacle',
              'FallingWithPS',
              'FallingBackSC',
              'Syncope',
              'FallingLeft']

# Import MatLab data as a dictionary
mat = scipy.loadmat('../data/UniMiB-SHAR/data/full_data.mat')

# Get numpy arrays containing data
data = mat['full_data']

total = 0
for p in range(len(data)):
  if p == 19:
    continue
  accel_data = data[p][0][0][0]
  a = 2
  activity = accel_data[a]
  for t in range(len(activity)):
    trial = activity[t][0]
    m = trial[5]
    print('person {} trial {} len {}'.format(p, t, len(m)))
    total += len(m)

avg = total / 58
print(avg)