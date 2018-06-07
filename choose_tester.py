import choose

data = []
person = [0, 1]
for i in range(5):
  data.append(person)

trials = choose.get_all_trials(data)
print 'all_trials:', trials
pairs = choose.get_pairs(trials)
print 'combinations:', pairs

output = choose.process(pairs)
for o in output:
  print o