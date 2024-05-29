
import os
import cv2
import numpy as np



#return train, val, test-seen, and test-unseen sets
def load_CXR_data():

    #relative path to training data
    train_path = "data/train"
    valid_path = "data/valid"
    test_path = "data/test-seen"
    test_path_unseen = "data/test-unseen"

    #size and shape of images
    image_size = (244, 244)
    image_shape = (244, 244, 3)

    train_x = []
    train_y = []

    val_x = []
    val_y = []

    test_seen_x = []
    test_seen_y = []

    test_unseen_x = []
    test_unseen_y = []

    # LOAD TRAIN DATA
    for file in os.listdir(train_path + "/covid"):
        
        image = cv2.imread(train_path + "/covid/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        train_x.append(image)
        train_y.append([1, 0])

    for file in os.listdir(train_path + "/pneumonia"):
        
        image = cv2.imread(train_path + "/pneumonia/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        train_x.append(image)
        train_y.append([0, 1])

    train_y = np.asarray(train_y).reshape(-1, 2)
    train_x = np.asarray(train_x)
    # ----------------

    # LOAD VAL DATA
    for file in os.listdir(valid_path + "/covid"):
        
        image = cv2.imread(valid_path + "/covid/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        val_x.append(image)
        val_y.append([1, 0])

    for file in os.listdir(valid_path + "/pneumonia"):
        
        image = cv2.imread(valid_path + "/pneumonia/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        val_x.append(image)
        val_y.append([0, 1])

    val_y = np.asarray(val_y).reshape(-1, 2)
    val_x = np.asarray(val_x)
    # ----------------


    # LOAD TEST-SEEN DATA
    for file in os.listdir(test_path + "/covid"):
        
        image = cv2.imread(test_path + "/covid/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        test_seen_x.append(image)
        test_seen_y.append([1, 0])

    for file in os.listdir(test_path + "/pneumonia"):
        
        image = cv2.imread(test_path + "/pneumonia/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        test_seen_x.append(image)
        test_seen_y.append([0, 1])

    test_seen_y = np.asarray(test_seen_y).reshape(-1, 2)
    test_seen_x = np.asarray(test_seen_x)
    # ----------------

    # LOAD TEST-UNSEEN DATA
    for file in os.listdir(test_path_unseen + "/covid"):
        
        image = cv2.imread(test_path_unseen + "/covid/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        test_unseen_x.append(image)
        test_unseen_y.append([1, 0])

    for file in os.listdir(test_path_unseen + "/pneumonia"):
        
        image = cv2.imread(test_path_unseen + "/pneumonia/" + file)
        image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        test_unseen_x.append(image)
        test_unseen_y.append([0, 1])

    test_unseen_y = np.asarray(test_unseen_y).reshape(-1, 2)
    test_unseen_x = np.asarray(test_unseen_x)
    # ----------------

    return train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y

 
    