import scipy.io as sio
import scipy.signal
from scipy import signal
import numpy as np
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from math import modf

DT = 0.02
FS = 1/DT
NUM_FEATURES = 7


def get_mean(sig):
  return np.mean(sig)

def get_std(sig):
  return np.std(sig)

def get_variance(sig):
  return get_std(sig)**2

def get_mad(sig):
  m = get_mean(sig)
  sig = [ abs(x - m) for x in sig ]
  return get_mean(sig)

def get_iqr(sig):
  q75, q25 = np.percentile(sig, [75 ,25])
  return q75 - q25

def get_energy(sig):
  # Not sure what nperseg should be here
  f_welch, S_xx_welch = scipy.signal.welch(sig, fs=FS, nperseg=len(sig)/2)
  df_welch = f_welch[1] - f_welch[0]
  dt = 1/FS
  f_fft = np.fft.fftfreq(len(sig), d=dt)
  df_fft = f_fft[1] - f_fft[0]
  E_welch = (1. / dt) * (df_welch / df_fft) * np.sum(S_xx_welch)
  return E_welch

def get_power(sig):
  # Not sure what nperseg should be here
  f_welch, S_xx_welch = scipy.signal.welch(sig, fs=FS, nperseg=len(sig)/2)
  df_welch = f_welch[1] - f_welch[0]
  P_welch = np.sum(S_xx_welch) * df_welch
  return P_welch

def get_feature_vector(sig):
  mean     = get_mean(sig)
  std      = get_std(sig)
  variance = get_variance(sig)
  mad      = get_mad(sig)
  iqr      = get_iqr(sig)
  energy   = get_energy(sig)
  power    = get_power(sig)
  return np.array([mean, std, variance, mad, iqr, energy, power])

def mag(sig):
  return np.linalg.norm(sig)

def coherence(sig1, sig2):
  coherence = scipy.signal.coherence(sig1, sig2
    , FS    # I think this may be wrong because when we compute coherence its between two coherence windows containing samples, so should it be DT or window length? Also not sure how it affects the math at all
    , nperseg=len(sig1)/2) # Not sure what nperseg should be
  return coherence

def N_signal(sig1, sig2, phi_max):
  f, C_xy = coherence(sig1, sig2)
  f[f < 10]
  # print 'f:', f
  # print 'C_xy:', C_xy
  C_xy = C_xy[:len(f)]
  # print 'trapz:', np.trapz(C_xy)
  # print 'sum:', np.sum(C_xy)
  return 1/float(phi_max) * np.sum(C_xy)

def split(sig, w):
  num_windows = float(len(sig)) / w
  dec, i = modf(num_windows)
  # print 'num_windows:', num_windows
  if num_windows != int(num_windows):
    cutoff = dec * w
    last = int(round(-1*cutoff))
    sig = sig[:last]
    num_windows = i
  # print 'num_windows:', num_windows
  return np.split(sig, num_windows)

def get_feature_matrix(sig, w):
  num_windows = len(sig) / w
  windows = split(sig, w)

  matrix = np.empty([num_windows, NUM_FEATURES])

  index = 0
  for window in windows:
    f = get_feature_vector(window)
    matrix[index] = f
    index += 1
  return matrix

def N_matrix(A, B, c, phi_max):
  num_windows = len(A)
  rows = num_windows - (c - 1)
  # print 'rows:', num_windows, '-(', c, '-1)=', rows
  matrix = np.empty([rows, NUM_FEATURES])

  for f in range(0, 7):
    A_feature = np.transpose(A)[f]
    B_feature = np.transpose(B)[f]
    # print 'A_feature:', A_feature
    # print 'B_feature:', B_feature

    for k in range(0, rows):
      A_samples = A_feature[k:k+c]
      B_samples = B_feature[k:k+c]
      # print 'A_samples:', A_samples
      # print 'B_samples:', B_samples
      cell = N_signal(A_samples, B_samples, phi_max)
      # print cell
      # print '\n'

      matrix[k][f] = cell

  return matrix

# def get_c(coherence_window):
#   return int((coherence_window) / (w * DT))


#################### TESTING ####################
'''

# import MatLab data as a dictionary
mat = sio.loadmat('./UniMiB-SHAR/data/full_data.mat')

data = mat['full_data']

person1 = 2
act_index1 = 0
trial_index1 = 0

person2 = 20
act_index2 = 1
trial_index2 = 1
sig1 = data[person1][0][0][0][act_index1][trial_index1][0][5]
sig2 = data[person1][0][0][0][act_index1][trial_index2][0][5]

short = min(len(sig1), len(sig2))
# short = 200 ###################################################### TESTING WITH SHORTER SIGNALS

sig1 = sig1[:short]
sig2 = sig2[:short]


print 'len(sig1):', len(sig1)
print 'len(sig2):', len(sig2)

w = 5
A = get_feature_matrix(sig1, w)
B = get_feature_matrix(sig2, w)

print 'A:', A
print 'B:', B

coherence_window = 4 # in seconds
phi_max = 10
c = int((coherence_window) / (w * DT))

print 'samples per window', w, 'samples'
print 'dt', DT, 'seconds'
print 'window length', w * DT, 'seconds'
print 'coherence window:', coherence_window, 'seconds'
print 'c:', c, 'windows'
print '\n\n\n\n'


coherence_matrix = N_matrix(A, B, c, phi_max)
print coherence_matrix

'''