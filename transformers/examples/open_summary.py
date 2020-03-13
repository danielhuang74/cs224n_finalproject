from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# STSB_on_CoLA = EventAccumulator('runs/STS-B_on_CoLA/events.out.tfevents.1584081577.nv6-2.113031.0')
# STSB_on_MRPC = EventAccumulator('runs/STS-B_on_MRPC')

# STSB_on_CoLA = EventAccumulator('runs/Feb27_19-33-45_cs224n-finalproject-nv6-new/events.out.tfevents.1582832025.cs224n-finalproject-nv6-new.74808.0')
# STSB_on_CoLA = EventAccumulator('runs/STS-B_baseline/events.out.tfevents.1584082439.nv6-2.117454.0')

STSB_on_CoLA = EventAccumulator('runs/STS-B_baseline_test/events.out.tfevents.1584087778.nv6-2.12277.0')




STSB_on_CoLA.Reload()
# STSB_on_MRPC.Reload()
# Show all tags in the log file
# print(STSB_on_CoLA)
# print()

# print(STSB_on_CoLA.Scalars('Accuracy'))
# print(STSB_on_CoLA.Scalars("loss"))


import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from tqdm import tqdm
from numpy import genfromtxt
import csv
from collections import defaultdict


steps = []
values = []
for wall_time, step, value in STSB_on_CoLA.Scalars("loss"):
    steps.append(step)
    values.append(value)

plt.plot(steps, values)
plt.show()


# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
#w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'