import choose

data = []
person = [0, 1]
for i in range(5):
  data.append(person)

trials = get_all_trials(data)
print 'all_trials:', trials
pairs = get_pairs(trials)
print 'combinations:', pairs

output = process(pairs)
for o in output:
  print o