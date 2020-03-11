import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from random import choices
percent = 10


train = -1
dev = -1

with open("train-v2.0big.json") as f:
    train = json.load(f)
f.close()
train_data = train["data"]

with open("dev-v2.0big.json") as f:
    dev = json.load(f)
f.close()
dev_data = dev["data"]

train_data = choices(train_data, k = math.ceil(len(train_data)/percent))
dev_data = choices(dev_data, k=math.ceil(len(dev_data)/percent))

train["data"] = train_data
dev["data"] = dev_data

print(len(train_data))
print(len(dev_data))


with open("train_10.json", "w") as f:
    json.dump(train, f)
f.close()

with open("dev_10.json", "w") as f:
    json.dump(dev, f)
f.close()