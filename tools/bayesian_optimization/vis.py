from hpbandster.api.results.util import logged_results_to_HB_result
from hpbandster.api.results.visualization import interactive_HB_plot

# Just to demonstrate, let's read in the logged runs rather than the returned Resu[t from HB.run
res = logged_results_to_HB_result('/mhome/chrabasp/EEG_Results/BO_Anomaly')

id2config = res.get_id2config_mapping()

print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))

incumbent_trajectory = res.get_incumbent_trajectory()

import matplotlib.pyplot as plt

interactive_HB_plot(incumbent_trajectory)

#
# plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
# plt.xlabel('wall clock time [s]')
# plt.ylabel('incumbent loss')
# plt.show()

