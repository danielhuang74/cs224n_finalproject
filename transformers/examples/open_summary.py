from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator('runs/Feb25_10-59-10_cs224n-finalproject-nv6-new/events.out.tfevents.1582628350.cs224n-finalproject-nv6-new.4660.0')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

print(event_acc.Scalars('loss'))

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
#w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))