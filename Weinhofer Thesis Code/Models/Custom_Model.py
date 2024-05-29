#Model definition for the custom convolutional model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

#much simpler version of model 1
def CustomModel(num_classes, input_shape):
    model = Sequential()

    #convolutional layer max pool and dropout
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    #convolutional layer max pool and dropout
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    #flatten model for upcoming FC layers
    model.add(Flatten())

    #output softmax layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model