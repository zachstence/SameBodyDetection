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

# given the accel_data for a particular person and a list of desired activities, extract all data
def getAccelData(accel_data):
  total = 0
  count = 0
  # for act_index in range(len(accel_data)):
  act_index = 2
  activity = accel_data[act_index]
  for trial_index in range(len(activity)):
    trial = activity[trial_index][0]
    x_accel          = trial[0]
    y_accel          = trial[1]
    z_accel          = trial[2]
    time_nanoseconds = trial[3]
    time_seconds     = trial[4]
    magnitude        = trial[5]
    total += len(magnitude)
    count += 1
    # At this point all fields are in variables for later processing
  print("average = {}".format(total / count))

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
mat = scipy.loadmat('./data/UniMiB-SHAR/data/full_data.mat')

# Get numpy arrays containing data
data = mat['full_data']

# Loop through data and extract all fields into variables
# for i in range(len(data)):
#   accel_data = data[i][0][0][0] # because each is wrapped in a 2D array
#   gender     = data[i][1][0][0]
#   age        = data[i][2][0][0]
#   height     = data[i][3][0][0]
#   weight     = data[i][4][0][0]
#   # for act_index in range(len(accel_data)):
#   act_index = 3
#   activity = accel_data[act_index]
#   print(len(activity))

p = 0
accel_data = data[p][0][0][0]
for act_index in range(len(accel_data)):
  activity = accel_data[act_index]
  print("{} : {}".format(ACTIVITIES[act_index], len(activity)))