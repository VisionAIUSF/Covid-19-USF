import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import gc
import os

from tensorflow.keras.datasets import fashion_mnist
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

learning_rate = 0.0001
batch_size = 64
num_classes = 10
results = []
epochs = 80
runs = 1

#download data
(x_train, y_train),(test_seen_x, test_seen_y) = fashion_mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
test_seen_y = keras.utils.to_categorical(test_seen_y, num_classes)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
test_seen_x = test_seen_x.reshape((test_seen_x.shape[0], 28, 28, 1))

x_train = x_train.astype("float32") / 255
test_seen_x = test_seen_x.astype("float32") / 255

x_valid = x_train[-5000:]
y_valid = y_train[-5000:]

x_train = x_train[:2500]
y_train = y_train[:2500]

test_unseen_x = test_seen_x.copy()
test_unseen_y = test_seen_y.copy()


print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", test_seen_x.shape)
print("y_test.shape = ", test_seen_y.shape)
print("x_valid.shape = ", x_valid.shape)
print("y_valid.shape = ", y_valid.shape)

print("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))

print(test_seen_y[0])





def add_gauss_noise(image, sigma, mean):
    row,col,ch= image.shape
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

for i, example in enumerate(test_unseen_x):
    test_unseen_x[i] = add_gauss_noise(example, 0.35, 0)


print("x_test_ood.shape = ", test_unseen_x.shape)
print("y_test_ood.shape = ", test_unseen_y.shape)



#main loop for each run of training with a random initialization of weights
for run in range(runs):
    print("******** Run Number: {} ********".format(run))

    #remove all weight files from weight folder. This makes training faster
    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)
    gc.collect()



    #define model 
    #model = RobustResnet18(10)
    #model.build(input_shape = (None,28,28,1))
    model = CustomModel(10, (28, 28, 1))
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
    model.fit(x=np.array(x_train, np.float32),
                y=np.array(y_train, np.float32),
                validation_data=(x_valid, y_valid),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                callbacks=checkpoint(Proposed_SWADS_Alt3, x_valid, y_valid))


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




