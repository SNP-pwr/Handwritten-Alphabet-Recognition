from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
from keras  import layers
import keras
import numpy as np
import cv2
import os, shutil, datetime, sys, time
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

ready = False
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '.' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s%s%s%s ...%s\r' % (bar, percents, '%  ', count, "/", total,  status))
    sys.stdout.flush()

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

    data = pd.read_csv(dat).astype('float32')

    class_mapping = {}

    X = data.drop('0',axis = 1) # axis=1 for dropping column
    y = data['0']
    X.head()
    y.head()

    y_full = y
    x_full = X.to_numpy().reshape(-1,28,28, 1)

    splitter = StratifiedShuffleSplit(n_splits=3,test_size=0.2)
    for train_ids, test_ids in splitter.split(x_full, y_full):
        X_train_full, y_train_full = x_full[train_ids], y_full[train_ids].to_numpy()
        X_test, y_test = x_full[test_ids], y_full[test_ids].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=t)

    return (X_train, y_train, X_valid, y_valid)



def buildModel(dat):
    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))

    model.add(Dense(26,activation ="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    return model

def trainModel(train_X, train_y, test_X, test_y, dat):
    classes = alphabets
    global model

    model = buildModel(dat)

    imageAug = Sequential([
        layers.RandomFlip("vertical"),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomRotation(0.3)])

    logdir = os.path.join(appFolder, "logs-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    total = len(train_X)
 
    for i in range(total):
        progress(i, total, status='Image augumentation ongoing...')
        train_X[i] = imageAug(train_X[i])
    print("\n")

    model.fit(train_X, train_y, epochs=50, batch_size=2048, callbacks=[tensorboard_callback])

    model.evaluate(test_X, test_y, batch_size=5)
    p = model.predict(test_X)
    cl=[]
    # print(p)
    # print(np.argmax(p[0]))
    # print(classes[np.argmax(p[0])])
    # print(np.argmax(p))
    for i in p:
        cl.append(np.argmax(i))
    conf_mat = tf.math.confusion_matrix(test_y, cl, num_classes=len(word_dict))
    # conf_mat = tf.math.confusion_matrix(test_y, model.predict_classes(test_X), num_classes=len(classes))
    print(conf_mat)

    model_json = model.to_json()
    with open("modelTest2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelTest2.h5")      #dawne model
    print("Saved model to disk")
    ready = True


def loadModel(modelPath):
    global model 
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model
    model.load_weights("modelTest2.h5")
    print("Loaded model from disk")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
    ready = True

def prepareSample():
    path = os.path.join(appFolder, "saved_files/image.png")
    try:
        imgArray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        newArray = cv2.resize(imgArray, (28, 28))
    except Exception as e:
        print('---------ERROR---------')

    sample = np.array(newArray).reshape(-1, imgSize, imgSize, 1)
    sample = sample/255.0
    return sample

def predict():
    new_X = prepareSample()
    # pred_y = model.predict_classes(new_X)
    pred = model.predict(new_X)
    #print(pred_y)
    # print(classes[pred_y[0]])
    print(max(pred[0]))
    print(pred[0])
    print(word_dict[np.argmax(pred[0])])
    # return classes[pred_y[0]]
    return word_dict[np.argmax(pred[0])]
