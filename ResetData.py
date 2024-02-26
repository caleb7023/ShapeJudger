#!/usr/bin/env python3

# Author: caleb7023

import numpy as np

NeuronWeights = np.zeros((128, 128), int)

np.save("./Data/NeuronWeights", NeuronWeights)

np.save("./Data/AccuaryList", np.array([]))

with open("./Data/Times", "w") as f:
    f.write("0")

with open("./Data/TotalFails", "w") as f:
    f.write("0")

print("Reset success.")