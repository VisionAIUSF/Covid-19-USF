from numpy.random import seed
import random as ran
import tensorflow as tf
import os
import numpy as np

seeds = [63528,30270,1186,47466,13938,27248,23050,32591,70485,44794,87752,67208,48357,41003,44268,55533,54862,59718,78523,69827,33651,12194,56602]


def setSeed(seed):
    newSeed = int(seed)

    ran.seed(newSeed)
    seed(newSeed)
    tf.random.set_seed(newSeed)

    session_conf = tf.compat.v1.ConfigProto()

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'


#returns the value of a gaussian at location x with standard deviation sigma
def gaussian (x, sigma):
    return (np.exp(-(x**2/(2*sigma*sigma)))*(1/(np.sqrt(2*np.pi)*sigma)))



#computes the validation loss on some model
def model_validation_loss(model, val_x, val_y):
    y_pred = model.predict(val_x, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(val_y, y_pred).numpy()
    return val_loss



