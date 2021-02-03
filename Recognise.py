import numpy as np
import cv2

from tensorflow import keras


def preprocessing(c):
    c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

    return c


def init(chars):
    dictionary = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    finalString = ''

    model = keras.models.load_model('myModel')

    for c in chars:
        c = np.asarray(c)
        c = cv2.resize(c, (32, 32))
        c = c / 255
        c = c.reshape(1, 32, 32, 1)

        classIndex = int(model.predict_classes(c))
        title = (dictionary[classIndex])
        finalString += title.strip("'[]")

    return finalString
