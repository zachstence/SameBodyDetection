import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

# Load data
mat = sio.loadmat('../data/UniMiB-SHAR/data/full_data.mat')
data = mat['full_data']

# Specify person index
person = 0
accel_data = data[person][0][0][0] # because each is wrapped in a 2D array

# Specify activity index (walking)
act_index = 2
activity = accel_data[act_index]

# Specify trial index
trial_index = 0
trial = activity[trial_index][0]

# Get all data fields
x_accel          = trial[0]
y_accel          = trial[1]
z_accel          = trial[2]
time_nanoseconds = trial[3]
time_seconds     = trial[4]
magnitude        = trial[5]

# Shorten data
t = 400
time_seconds = time_seconds[:t]
x_accel   = x_accel[:t]
y_accel   = y_accel[:t]
z_accel   = z_accel[:t]
magnitude = magnitude[:t]

# plt.style.use('dark_background')
labels = ['x', 'y', 'z', 'magnitude']
graphs = [x_accel, y_accel, z_accel, magnitude]
colors = ['r', 'y', 'm', 'c']

fig, axes = plt.subplots(4, sharex=True)

# Plot x, y, z and magnitude with colors and labels
for i, axis in enumerate(axes):
  axis.plot(time_seconds, graphs[i], colors[i])
  axis.set_ylabel(labels[i])
  axis.yaxis.set_label_position('right')

# Add labels
a = 0.5
b = 0.03
fig.text(a, b, 'Time (seconds)', ha='center')
fig.text(b, a, 'Acceleration (m/s^2)', va='center', rotation='vertical')
plt.suptitle('Acceleration Data from Walking\n(time vs. acceleration)')

# Save figure
plt.savefig('example_light.png', dpi=600, transparent=False)