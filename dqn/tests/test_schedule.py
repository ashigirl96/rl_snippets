"""Test Schedule"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from dqn.configs import default_config
from dqn.utility import PiecewiseSchedule


_config = default_config()

num_iterations = float(_config.max_total_step_size) / 4.0
lr_multiplier = 1.0
_lr_schedule = PiecewiseSchedule([
  (0, 1e-4 * lr_multiplier),
  (num_iterations / 10, 1e-4 * lr_multiplier),
  (num_iterations / 2, 5e-5 * lr_multiplier),
], outside_value=5e-5 * lr_multiplier)

ts = np.arange(0., _config.max_total_step_size)
values = []
for t in ts:
  values.append(_lr_schedule.value(t))

values = np.asarray(values)
print(values[0:10])
plt.plot(ts, values)
plt.show()

