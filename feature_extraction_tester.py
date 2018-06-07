import feature_extraction

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