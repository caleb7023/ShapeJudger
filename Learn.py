#!/usr/bin/env python3

# Author: caleb7023

import numpy as np

# Either to show the lean progress or to debug
import cv2

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from random import getrandbits, randrange


def Sqrt(n:int) -> float:
    return n ** 0.5



# Calc the distance between 2 Pos
def Distance(Pos1:tuple, Pos2:tuple) -> float:
    HeightX = Pos1[0] - Pos2[0]
    WidthY  = Pos1[1] - Pos2[1]
    return Sqrt(HeightX * HeightX + WidthY * WidthY) # looks sexy



def CreateRandomShapeImg():

    # Create a img with bool
    Img = np.zeros((128 , 128) , bool)

    # Pos1
    Pos1 = (randrange(0, 100), randrange(0, 100))

    # Pos2, will be bigger than Pos1 value
    Pos2 = (randrange(Pos1[0] + 27, 127), randrange(Pos1[1] + 27, 127))

    # if getrandbits(1) returns 1, its gonna create img of a rectangle.
    # else its gonna return img of a ellipse.
    Shape = getrandbits(1)


    if Shape:

        # Fill inside of the rectangle
        Img[Pos1[0] : Pos2[0],
            Pos1[1] : Pos2[1]] = True


    else:
        # The center of the ellipse
        EllipseCenter = ((Pos1[0] + Pos2[0]) * 0.5, (Pos1[1] + Pos2[1]) * 0.5)

        # Height and width ratio
        WidthRatio = ((Pos2[0] - Pos1[0]) / (Pos2[1] - Pos1[1]))

        # The radius of the ellipse of the x axis
        Radius = EllipseCenter[0] - Pos1[0]

        # Check is the each pixel in the ellipse or not
        for PosX in range(127):

            for PosY in range(127):
                
                # Scale the y axis usin WidthRatio and check is the pixel in the radius
                if Distance(EllipseCenter, (PosX, EllipseCenter[1] + (PosY - EllipseCenter[1]) * WidthRatio)) < Radius:

                    # Change the pixel False to True
                    Img[PosX, PosY] = True
    
    for i in range(randrange(0, 3)):

        Img = np.rot90(Img)

    return Img, Shape



def learn():

    NeuronWeights = np.load("./Data/NeuronWeights.npy")
    
    with open("./Data/Times", "r") as f:
        Times = int(f.read())

    with open("./Data/TotalFails", "r") as f:
        TotalFails = int(f.read())

    AccuaryList = np.load("./Data/AccuaryList.npy")

    while True:

        Times += 1

        Fails = 0
        
        for i in range(5000):
            # If shape is 1, it means the shape is rectangle.
            # If shape is 0, it means the shape is ellipse.

            Img, Shape = CreateRandomShapeImg()
            Rectangle = 6 < np.sum(Img * NeuronWeights)

            if Rectangle != Shape:
                Fails += 1
                # The shape was rectangle
                if Shape:
                    NeuronWeights += Img
                else:
                    NeuronWeights -= Img

            cv2.imshow("Img", np.uint8(Img * 255))
            a = cv2.waitKeyEx(1)
        
        TotalFails += Fails

        AccuaryList = np.append(AccuaryList, Fails * 0.0002)

        ##################
        # Render Heatmap #
        ##################
            
        fig = plt.figure()
                
        sns.heatmap(NeuronWeights, cmap="viridis")
        fig.canvas.draw()
        NeuronWeightsHeatMap = np.array(fig.canvas.renderer.buffer_rgba())

        fig.clf()
        plt.close()
        del fig
        gc.collect()

        NeuronWeightsHeatMap = cv2.cvtColor(NeuronWeightsHeatMap, cv2.COLOR_RGBA2BGR)

        cv2.imshow("Neuron weights heat map", NeuronWeightsHeatMap)

        #####################
        # Render Line Graph #
        #####################

        
        fig = plt.figure()
                
        plt.plot(AccuaryList, scaley=False)
        fig.canvas.draw()
        AccuaryLineGraph = np.array(fig.canvas.renderer.buffer_rgba())

        fig.clf()
        plt.close()
        del fig
        gc.collect()

        AccuaryLineGraph = cv2.cvtColor(AccuaryLineGraph, cv2.COLOR_RGBA2BGR)

        cv2.imshow("Accuary Line Graph", AccuaryLineGraph)

        ##############
        # Save Datas #
        ##############
        
        np.save("./Data/NeuronWeights.npy", NeuronWeights)
        
        np.save("./Data/AccuaryList.npy", AccuaryList)

        with open("./Data/Times", "w") as f:
            f.write(str(Times))

        with open("./Data/TotalFails", "w") as f:
            f.write(str(TotalFails))

        ###############
        # Print Infos #
        ###############

        print("Times:{0}, AvgAccuracy:{1}, Accuracy:{2}".format(Times, round(TotalFails / ((Times) * 5000), 3), round(Fails * 0.0002, 3)))

if __name__ == "__main__":
    learn()