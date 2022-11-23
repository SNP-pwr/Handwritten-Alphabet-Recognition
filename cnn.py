from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras  import layers
import keras
import numpy as np
import cv2
import os, shutil, datetime
import random
from keras.models import model_from_json
from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


appFolder = os.path.dirname(os.path.abspath(__file__))


imgSize = 28
alphabets = []

trainData = []
train_X = []
train_y = []
test_X = []
test_y = []

def mostFrequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def createDataSet(dat, t):
    #classes = os.listdir(dat)
    tempData = []
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    df = pd.read_csv(dat)

    class_mapping = {}
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for i in range(len(alphabets)):
        class_mapping[i] = alphabets[i]

    df['class'].map(class_mapping).unique()

    y_full = df.pop('class')
    x_full = df.to_numpy().reshape(-1,28,28, 1)

    splitter = StratifiedShuffleSplit(n_splits=3,test_size=0.2)
    for train_ids, test_ids in splitter.split(x_full, y_full):
        X_train_full, y_train_full = x_full[train_ids], y_full[train_ids].to_numpy()
        X_test, y_test = x_full[test_ids], y_full[test_ids].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=t)

    return (X_train, y_train, X_valid, y_valid)



def buildModel(dat):
    
    model = Sequential()

    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(2,2))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(len(classes), activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    return model

def trainModel(train_X, train_y, test_X, test_y, dat):
    classes = alphabets
    global model

    model = buildModel(dat)

    logdir = os.path.join(appFolder, "logs-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(train_X, train_y, epochs=20, batch_size=64, callbacks=[tensorboard_callback])

    model.evaluate(test_X, test_y, batch_size=5)
    p = model.predict(test_X)
    cl=[]
    # print(p)
    # print(np.argmax(p[0]))
    # print(classes[np.argmax(p[0])])
    # print(np.argmax(p))
    for i in p:
        cl.append(np.argmax(i))
    conf_mat = keras.math.confusion_matrix(test_y, cl, num_classes=len(classes))
    # conf_mat = tf.math.confusion_matrix(test_y, model.predict_classes(test_X), num_classes=len(classes))
    print(conf_mat)

    model_json = model.to_json()
    with open("modelTest2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelTest2.h5")      #dawne model
    print("Saved model to disk")


def loadModel(modelPath):
    global model 
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

def prepareSample():
    path = os.path.join(appFolder, "saved_files/")
    try:
        imgArray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        newArray = cv2.resize(imgArray, (28, 28))
    except Exception as e:
        print('---------ERROR---------')

    sample = np.array(newArray).reshape(-1, imgSize, imgSize, 1)
    sample = sample/255.0
    return sample

def predict(new_X, dat):
    classes = os.listdir(dat)
    # pred_y = model.predict_classes(new_X)
    pred = model.predict(new_X)
    #print(pred_y)
    # print(classes[pred_y[0]])
    print(max(pred[0]))
    print(pred[0])
    # return classes[pred_y[0]]
    return classes[np.argmax(pred[0])]
