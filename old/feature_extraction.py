from scipy import signal
import numpy as np
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
  f_welch, S_xx_welch = signal.welch(sig, fs=FS, nperseg=len(sig)/2)
  df_welch = f_welch[1] - f_welch[0]
  dt = 1/FS
  f_fft = np.fft.fftfreq(len(sig), d=dt)
  df_fft = f_fft[1] - f_fft[0]
  E_welch = (1. / dt) * (df_welch / df_fft) * np.sum(S_xx_welch)
  return E_welch

def get_power(sig):
  # Not sure what nperseg should be here
  f_welch, S_xx_welch = signal.welch(sig, fs=FS, nperseg=len(sig)/2)
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
  coherence = signal.coherence(sig1, sig2
    , FS    # I think this may be wrong because when we compute coherence its between two coherence windows containing samples, so should it be DT or window length? Also not sure how it affects the math at all
    , nperseg=len(sig1)/2) # Not sure what nperseg should be
  return coherence

def N_signal(sig1, sig2, phi_max):
  f, C_xy = coherence(sig1, sig2)
  f[f < 10]
  C_xy = C_xy[:len(f)]
  return 1/float(phi_max) * np.sum(C_xy)

def split(sig, w):
  num_windows = float(len(sig)) / w
  dec, i = modf(num_windows)
  if num_windows != int(num_windows):
    cutoff = dec * w
    last = int(round(-1*cutoff))
    sig = sig[:last]
    num_windows = i
  return np.split(sig, num_windows)

def get_feature_matrix(sig, w):
  num_windows = int(len(sig) / w)
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
  matrix = np.empty([rows, NUM_FEATURES])

  for f in range(0, 7):
    A_feature = np.transpose(A)[f]
    B_feature = np.transpose(B)[f]

    for k in range(0, rows):
      A_samples = A_feature[k:k+c]
      B_samples = B_feature[k:k+c]
      cell = N_signal(A_samples, B_samples, phi_max)

      matrix[k][f] = cell

  return matrix

def get_c(w, coherence_window):
  c = int((coherence_window) / (w * DT))
  return c