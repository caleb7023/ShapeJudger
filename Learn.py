#!/usr/bin/env python3

# Author: caleb7023

import cupy as cp

# Either to show the lean progress or to debug

import time

from random import getrandbits, randrange



# Generate a array of X and Y pos of 128x128 array
Size = cp.arange(0, 128)
PosXArray, PosYArray = cp.meshgrid(Size, Size)
PosXArray = cp.array(cp.float64(PosXArray.get()))
PosYArray = cp.array(cp.float64(PosYArray.get()))
del Size



# Render ellipse.
# The Pos1 should be smaller than Pos2.
def CreateEllipse(Pos1:tuple, Pos2:tuple) -> cp.array:

    global PosXArray, PosYArray

    # Calc the center of the ellipse
    EllipseCenter = ((Pos1[0] + Pos2[0]) * 0.5, (Pos1[1] + Pos2[1]) * 0.5)

    # Width ratio
    WidthRatio = ((Pos2[0] - Pos1[0]) / (Pos2[1] - Pos1[1]))

    # The radius of the x axis
    XRadius = (Pos2[0] - Pos1[0]) * 0.5

    TempPosXArray, TempPosYArray = cp.array(PosXArray), cp.array(PosYArray)

    TempPosXArray -= EllipseCenter[0]
    TempPosYArray -= EllipseCenter[1]

    # To create the ellipse as circle
    TempPosYArray *= WidthRatio

    # Calc all the distance from Pos1
    DistanceArray = cp.sqrt(cp.square(TempPosXArray) + cp.square(TempPosYArray))

    # If the distance were smaller than the X radius, the aug gonna be True
    return DistanceArray < XRadius



def CreateRandomShapeImg() -> cp.array:

    # Create a img with bool
    Img = cp.zeros((128 , 128) , bool)

    # Pos1
    Pos1 = (randrange(0, 100), randrange(0, 100))

    # Pos2, will be bigger than Pos1 value
    Pos2 = (randrange(Pos1[0] + 27, 127), randrange(Pos1[1] + 27, 127))

    # if getrandbits(1) returns 1, its gonna create img of a rectangle.
    # else its gonna return img of a ellipse.
    Shape = getrandbits(1)

    if Shape:
        # Get an img of a rectangle
        Img[Pos1[0] : Pos2[0],
            Pos1[1] : Pos2[1]] = True
    else:
        # Get an img of an ellipse
        Img = CreateEllipse(Pos1, Pos2)

    for i in range(randrange(0, 3)):
        Img = cp.rot90(Img)

    return Img, Shape



def learn(SaveToDisk:bool = True):

    NeuronWeights = cp.load("./Data/NeuronWeights.npy")

    AccuaryList = cp.load("./Data/AccuaryList.npy")
    
    with open("./Data/Terms", "r") as f:
        Terms = int(f.read())

    with open("./Data/TotalFails", "r") as f:
        TotalFails = int(f.read())

    while True:

        StartTime = time.time()

        Fails = 0
        
        for i in range(50000):

            Terms += 1
            
            # If shape is 1, it means the shape is rectangle.
            # If shape is 0, it means the shape is ellipse.

            Img, Shape = CreateRandomShapeImg()
            Rectangle = 0 < cp.sum(Img * NeuronWeights)
            if Rectangle != Shape:
                Fails += 1

                # If the shape is rectangle
                if Shape:
                    NeuronWeights += Img
                # If the shape is ellipse
                else:
                    NeuronWeights -= Img
            

        TotalFails += Fails

        AccuaryList = cp.append(AccuaryList, Fails * 0.00002)

        ##########################
        # Save datas to the disk #
        ##########################
        
        if SaveToDisk:

            cp.save("./Data/NeuronWeights", NeuronWeights)
            
            cp.save("./Data/AccuaryList", AccuaryList)

            with open("./Data/Terms", "w") as f:
                f.write(str(Terms))

            with open("./Data/TotalFails", "w") as f:
                f.write(str(TotalFails))

        ###############
        # Print infos #
        ###############

        print("Terms:{0}, TotalFails:{1}, Accuracy:{2}%, SexPer10000Time:{3}".format(Terms,
                                                                                     TotalFails,
                                                                                     round(Fails * 0.002, 1),
                                                                                     round((time.time() - StartTime) * 0.2, 3)))

if __name__ == "__main__":
    learn(SaveToDisk=True)



# Im out of brain cells rn