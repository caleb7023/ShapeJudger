#!/usr/bin/env python3

# Author: caleb7023

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


NeuronWeights = np.load("./Data/NeuronWeights.npy")

AccuaryList = np.load("./Data/AccuaryList.npy")

fig1 = plt.figure()

fig2 = plt.figure()

##################
# Render Heatmap #
##################
        
sns.heatmap(NeuronWeights, cmap="viridis")
fig1.canvas.draw()

#####################
# Render Line Graph #
#####################
        
plt.plot(AccuaryList, scaley=False)
fig2.canvas.draw()

while True:

    plt.pause(0.05)