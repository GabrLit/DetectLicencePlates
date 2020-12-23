import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.keras.optimizers import Adam
import pickle


def recognise():
    TEST_RATIO = 0.2
    VALIDATION_RATIO = 0.2

    images = []
    labels = []
    labelsDir = sorted(os.listdir('dataset_characters'))
    noOfLabels = len(labelsDir)

    print("Importing Dataset...")
    for l in range(0, noOfLabels):
        imgdir = (os.listdir("dataset_characters/{}".format(l)))
        for image in imgdir:
            path = ("dataset_characters/{}/{}".format(l, image))
            currentImg = cv2.imread(path)
            currentImg = cv2.resize(currentImg, (32, 32))
            images.append(currentImg)
            labels.append(l)
        print(l, end=" ")
    print()
    print("Import Completed...")

    images = np.array(images)
    labels = np.array(labels)

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_RATIO)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_RATIO)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_validation.shape)

    noOfSamples = []
    for l in sorted(list(dict.fromkeys(labels))):
        noOfSamples.append(len(np.where(y_train == l)[0]))

    # print(noOfSamples)

    plt.figure(figsize=(10, 5))
    plt.bar(range(0, noOfLabels), noOfSamples)
    plt.title("Number of images for each class")
    plt.xlabel("label ID")
    plt.ylabel("number of images")
    plt.show()

    # img = X_train[30]
    # img= cv2.resize(img, (300,300))
    # print(X_train[30].shape)
    # cv2.imshow("Preprocessed", img)
    # cv2.waitKey(0)

    X_train = np.array(list(map(preprocessing, X_train)))
    X_test = np.array(list(map(preprocessing, X_test)))
    X_validation = np.array(list(map(preprocessing, X_validation)))

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)

    dataGen.fit(X_train)
    y_train = to_categorical(y_train, noOfLabels)
    y_test = to_categorical(y_test, noOfLabels)
    y_validation = to_categorical(y_validation, noOfLabels)

    # print(y_train)
    # print(X_test.shape)
    # print(X_validation.shape)

    model = myModel(noOfLabels)
    print(model.summary())

    BATCH_SIZE = 32
    EPOCHS = 50
    STEPS_PER_EPOCH = X_train.shape[0]//BATCH_SIZE

    history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  epochs=EPOCHS,
                                  validation_data=(X_validation, y_validation),
                                  shuffle=1)

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('epoch')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('accuracy')
    plt.xlabel('epoch')

    plt.show()
    score= model.evaluate(X_test, y_test, verbose=0)
    print('Test Score = ', score[0])
    print('Accuracy = ', score[1])

    model.save("myModel")

    #pickle_out=open("model_trained.p","wb")
    #pickle.dump((model, pickle_out))
    #pickle_out.close()

    return 0


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    return img


def myModel(noOfLabels):
    noOfFilters = 80
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 1000

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfLabels, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


recognise()
