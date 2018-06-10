import feature_extraction as fe
import scipy.io as sio

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


print('len(sig1): {}'.format(len(sig1)))
print('len(sig2): {}'.format(len(sig2)))

w = 5
A = fe.get_feature_matrix(sig1, w)
B = fe.get_feature_matrix(sig2, w)

print('A: {}'.format(A))
print('B: {}'.format(B))

coherence_window = 4 # in seconds
phi_max = 10
c = int((coherence_window) / (w * fe.DT))

print('samples per window {} samples'.format(w))
print('dt {} seconds'.format(fe.DT))
print('window length {} seconds'.format(w * fe.DT))
print('coherence window: {} seconds'.format(coherence_window))
print('c: {} windows'.format(c))
print('\n\n\n\n')

coherence_matrix = fe.N_matrix(A, B, c, phi_max)
print(coherence_matrix)