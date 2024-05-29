import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import os

from CXR_data_loader import load_CXR_data
from utility import setSeed
from utility import model_validation_loss

from SWAD_algos import Original_SWAD
from SWAD_algos import Proposed_SWADS
from SWAD_algos import Proposed_SWADS_Alt1
from SWAD_algos import Proposed_SWADS_Alt2
from SWAD_algos import Proposed_SWADS_Alt3

from SWAD_utility import AverageWeights
from Callbacks import SWAD_callback
from Callbacks import checkpoint

from Models.Resnet_18 import ResNet18
from Models.Resnet_9 import ResNet9
from Models.Custom_Model import CustomModel
from Models.Robust_Resnet_18 import RobustResnet18
from tensorflow.keras.applications.densenet import DenseNet121


print("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))


#hyperparameters
NS = 6
NE = 6
r = 1.2
N = 5

#setup and global vars
new_algo_widths = []
original_algo_widths = []
results = []
learning_rate = 0.0001
batch_size = 32
epochs = 50
runs = 1




#load in training data
train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y = load_CXR_data()



#main loop for each run of training with a random initialization of weights
for run in range(runs):
    print("******** Run Number: {} ********".format(run))

    #remove all weight files from weight folder. This makes training faster
    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)
    gc.collect()



    #define model 
    #model = RobustResnet18(2)
    #model.build(input_shape = (None,244,244,3))
    model = CustomModel(2, (244,244,3))
    #model = DenseNet121(input_shape=(244,244,3), classes=2, weights=None)

    #define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

    #compile model with accuracy metric
    model.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=['accuracy'])



    #train the model
    #For proposed SWAD-S : callbacks=SWAD_callback(Proposed_SWADS, val_x, val_y))
    #For original SWAD : callbacks=SWAD_callback(Original_SWAD, val_x, val_y))         callbacks=SWAD_callback(Proposed_SWADS_Alt1, val_x, val_y)
    model.fit(x=np.array(train_x, np.float32),
                y=np.array(train_y, np.float32),
                validation_data=(val_x, val_y),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                callbacks=checkpoint(Proposed_SWADS_Alt1, val_x, val_y))


    #Evaluate model on seen and unseen data
    scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
    print('Test loss seen:', scores[0])
    print('Test accuracy seen:', scores[1])

    scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
    print('Test loss unseen:', scores_unseen[0])
    print('Test accuracy unseen:', scores_unseen[1])

    results.append([scores[1], scores_unseen[1]])








#save results to file for import into excel
df = pd.DataFrame(results)
df.to_csv('multirun_results.csv')

print("\n\n")
#final result print out
for i, x in enumerate(results):
    print("Run: {}, Accuracy-Seen: {}".format(i, x[0]))
    print("\nRun: {}, Accuracy-unSeen: {}\n\n".format(i, x[1]))


#16.5