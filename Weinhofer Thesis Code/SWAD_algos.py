#SWAD algorithms for finding start and end iterations
#Written by Brandon Weinhofer

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

from SWAD_utility import oneNGreaterThan
from SWAD_utility import AveragedLoss

import scipy.ndimage as scimage
from utility import gaussian


#Proposed SWAD-S algorithm
#inputs: Validation loss vector, patience parameter N
#outputs: TS (starting iteration), TE (end iteration)
def Proposed_SWADS(val_loss, N = 5):

    #index of the min loss value for starting point
    min_index = val_loss.index(min(val_loss))
    TS = min_index
    TE = min_index + 1


    #Handle finding starting iteration
    if N < TS:
        while oneNGreaterThan(val_loss, N, TS, last=True) and TS != 0:
            TS = TS - 1
    else:
        TS = 0
    

    #Handle finding ending iteration
    if N < len(val_loss)-TE:
        while oneNGreaterThan(val_loss, N, TE, last=False) and TE != len(val_loss)-1:
            TE = TE + 1
    else:
        TE = len(val_loss)-1
    

    return TS, TE


def Proposed_SWADS_Alt1(val_loss, N=20, r=0.96):
    min_index = val_loss.index(min(val_loss))
    TS = min_index
    TE = min_index + 1
    threshold = val_loss[min_index] * r

    #Handle finding starting iteration
    if N < TS:
        while AveragedLoss(val_loss, N, TS, True) >= val_loss[TS]*r and TS != 0:
            TS = TS - 1
    else:
        TS = 0


    #Handle finding ending iteration
    if N < len(val_loss)-TE:
        while AveragedLoss(val_loss, N, TE, last=False) >= val_loss[TE]*r and TE != len(val_loss)-1:
            TE = TE + 1
    else:
        TE = len(val_loss)-1
    
    return TS, TE


def Proposed_SWADS_Alt2(val_loss, N=20, r=0.96):
    t = np.linspace(-len(val_loss)/2, len(val_loss)/2, len(val_loss))
    ts, te = Proposed_SWADS_Alt1(list(scimage.convolve(val_loss, gaussian(t, 8))), N, r)

    return ts, te


def Proposed_SWADS_Alt3(val_loss, N=20, r=0.96):
    
    ts, _ = Proposed_SWADS_Alt1(val_loss)
    _, te = Original_SWAD(val_loss)

    if te <= ts:
        te = ts + 1

    return ts, te


#return a start and end point from the original swad algorithm
#inputs are NS (optimum patience), NE(overfit patience), and r (ratio)
def Original_SWAD(val_loss, NS=6, NE=6, r=1.2):
    ts = 0
    te = len(val_loss)
    l = None

    for i in range(NS-1, len(val_loss)):
        
        min1 = math.inf
        for j in range(NE):
            if val_loss[i-j] < min1:
                min1 = val_loss[i-j]
        
        if l == None:
            
            min = math.inf
            for j in range(NS):
                if val_loss[i-j] < min:
                    min = val_loss[i-j]

            if val_loss[i-NS+1] == min:

                ts = i-NS+1
                sums = 0
                for j in range(NS):
                    sums = sums + val_loss[i-j]
                l = (r/NS)*sums
        
        elif l < min1:
            te = i-NE
            break
    return ts, te









