#utility functions for SWAD
#written by Brandon Weinhofer

import numpy as np
import math

#Helper function for new SWAD-S algorithm that returns true if one of the last
#N values in a list is greater than the currect value
def oneNGreaterThan(val_loss, N, curr_val, last):

    if last:
        if curr_val-N <= 0:
            return -1

        last_n_vals = val_loss[curr_val-N:curr_val]

    else:
        if len(val_loss)-curr_val-N <= 0:
            return -1
        
        last_n_vals = val_loss[curr_val+1:curr_val+N+1]


    last_vals_greater = [x for x in last_n_vals if x > val_loss[curr_val]]

    if len(last_vals_greater) > 0:
        return True
    else:
        return False




#returns the average loss over a pre determined range
def AveragedLoss(val_loss, N, curr_val, last):

    if last:
        if curr_val-N <= 0:
            return -1

        last_n_vals = val_loss[curr_val-N:curr_val]

    else:
        if len(val_loss)-curr_val-N <= 0:
            return -1
        
        last_n_vals = val_loss[curr_val+1:curr_val+N+1]

    return (sum(last_n_vals) / len(last_n_vals))



#Probaby could be re written
#averages weights in "Weights/" folder and returned averaged model parameters between two points (ts and te)
#max load is the maximum number of weights that should be loaded into memory at once
#as when I run locally I run into memory limitations
def AverageWeights(model, ts, te, max_load):

    curr = ts
    stop = te
    iterations = stop - curr
    current_averaged = 0

    whole_averages = int(iterations / max_load)
    remainder = iterations % max_load


    weight_set = []
    new_weights = list()

    folder_prefix = "Weights/weights_"

    #average each whole chunk of weights
    for i in range(whole_averages):

        weight_set.clear()
        new_weights.clear()

        for j in range(curr, curr+max_load):
            model.load_weights(folder_prefix + str(j) + ".weights.h5")
            weight_set.append(model.get_weights())
            curr += 1
        
        for weights_list_tuple in zip(*weight_set): 
                new_weights.append(
                    np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                )

        model.set_weights(new_weights)
        model.save_weights(folder_prefix + str(current_averaged) + ".weights.h5")
        current_averaged += 1

    #average the remaining weights if there are any
    if remainder > 0:
        weight_set.clear()
        new_weights.clear()
        for i in range(curr, curr+remainder):
            model.load_weights(folder_prefix + str(i) + ".weights.h5")
            weight_set.append(model.get_weights())
            curr += 1

        for weights_list_tuple in zip(*weight_set): 
                    new_weights.append(
                        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                    )

        model.set_weights(new_weights)
        model.save_weights(folder_prefix + str(current_averaged) + ".weights.h5")
        current_averaged += 1

    #average the sub_averages
    weight_set.clear()
    new_weights.clear()
    for i in range(current_averaged):
        model.load_weights(folder_prefix + str(i) + ".weights.h5")
        weight_set.append(model.get_weights())   

    for weights_list_tuple in zip(*weight_set): 
                new_weights.append(
                    np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                )

    return new_weights



