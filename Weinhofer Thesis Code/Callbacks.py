import tensorflow as tf
import os
import gc
import numpy as np
import pandas as pd

import scipy.ndimage as scimage
from utility import model_validation_loss, gaussian
from SWAD_utility import AverageWeights





#checkpoint callback for "baseline" performance
#finds model weights from iteration with lowest validaion loss
class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self, SWAD_Version, val_x, val_y):
        self.min_loss = float("inf")
        self.opt_weight = None
        self.val_x = val_x
        self.val_y = val_y
        self.SWAD_Version = SWAD_Version

        #list to save loss curve
        self.loss_tracker = []

    def on_train_batch_end(self, epoch, logs=None):
        val_loss = model_validation_loss(self.model, self.val_x, self.val_y)
        self.loss_tracker.append(val_loss)

        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.opt_weight = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.opt_weight)

        ts, te = self.SWAD_Version(self.loss_tracker)
        print("GAP: {}".format(te-ts))

        df = pd.DataFrame(self.loss_tracker)
        df.to_csv('loss.csv')




#callback for SWAD algorithm
class SWAD_callback(tf.keras.callbacks.Callback):

    def __init__(self, SWAD_Version, val_x, val_y):
        self.loss_tracker = []
        self.weights_saved = 0
        self.new_weights = list()
        self.SWAD_Version = SWAD_Version
        self.val_x = val_x
        self.val_y = val_y

        #make sure weights folder exists and if it does make sure it is empty
        if not os.path.isdir('Weights/'):
            os.mkdir("Weights")
        else:
            weights_folder = os.listdir("Weights")
            for file in weights_folder:
                os.remove("Weights/"+file)
        

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

    def on_train_batch_end(self, batch, logs=None):
        val_loss = model_validation_loss(self.model, self.val_x, self.val_y)
        self.loss_tracker.append(val_loss)
        self.model.save_weights("Weights/weights_" + str(self.weights_saved) + ".weights.h5")
        self.weights_saved += 1

    def on_train_end(self, logs=None):
        #t = np.linspace(-len(self.loss_tracker)/2, len(self.loss_tracker)/2, len(self.loss_tracker))
        #ts, te = self.SWAD_Version(list(scimage.convolve(self.loss_tracker, gaussian(t, 8))))
        ts, te = self.SWAD_Version(self.loss_tracker)

        print("TS: {} TE: {} GAP: {}".format(ts, te, te-ts))
        self.new_weights = AverageWeights(self.model, ts, te, 200)

        #set model weights to new average
        if len(self.new_weights) > 0:
            print("\nSetting new model weights.\n")
            self.model.set_weights(self.new_weights)