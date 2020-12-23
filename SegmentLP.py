import Settings
import cv2
import numpy as np

ASPECT_RATIO = 0.9
AREA_RATIO = 0.15
HEIGHT_RATIO_MIN = 0.75
HEIGHT_RATIO_MAX = 0.95


def init(licencePlate):
    segmented = []
    empty = np.zeros_like(licencePlate)

    LPGrayScale = cv2.cvtColor(licencePlate, cv2.COLOR_BGR2GRAY)
    LPBlurred = cv2.bilateralFilter(LPGrayScale, 19, 5, 5)
    LPBinary = cv2.adaptiveThreshold(LPBlurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    if Settings.showSteps["LPGrayScale"] == 1:
        cv2.imshow("LPGrayScale", LPGrayScale)
        cv2.waitKey(0)

    if Settings.showSteps["LPBinary"] == 1:
        cv2.imshow("LPBinary", LPBinary)
        cv2.waitKey(0)

    contours2, hierarchy = cv2.findContours(LPBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if Settings.showSteps["LPContours"] == 1:
        cv2.drawContours(empty, contours2, -1, (255, 255, 255), 1)  # drawing on an image
        cv2.imshow("LPContours", empty)
        cv2.waitKey(0)

    # Iterating through contours and searching for char candidates
    i = len(contours2) - 1
    for c in reversed(contours2):
        box = cv2.boundingRect(c)

        aspectRatio = box[2] / box[3]  # width/height
        areaRatio = cv2.contourArea(c) / float(box[2] * box[3])  # solidity
        heightRatio = box[3] / licencePlate.shape[0]  # aspect of character to plate height

        if aspectRatio > ASPECT_RATIO or areaRatio < AREA_RATIO or heightRatio < HEIGHT_RATIO_MIN or heightRatio > HEIGHT_RATIO_MAX or hierarchy[0][i][3] != -1:
            del contours2[i]
        i = i - 1

    # Sorting contours left-to-right
    if len(contours2) > 4:
        boundingBoxes = [cv2.boundingRect(c) for c in contours2]
        (contours2, boundingBoxes) = zip(*sorted(zip(contours2, boundingBoxes), key=lambda b: b[1][0], reverse=False))
    else:
        return 0

    i = 1
    chars = []
    for c in contours2:
        x, y, w, h = cv2.boundingRect(c)

        character = LPBinary[y:y + h, x:x + w]

        # scale up to 60 height
        r = 60 / character.shape[0]
        dim = (int(character.shape[1] * r), 60)
        character = cv2.resize(character, dim)

        paddedWidth = 80
        charWidth = character.shape[1]

        if charWidth % 2 == 0:
            pad = int((paddedWidth - charWidth) / 2)
            paddedChar = cv2.copyMakeBorder(character, 10, 10, pad, pad, cv2.BORDER_CONSTANT)
        else:
            pad = int((paddedWidth - charWidth) / 2)
            paddedChar = cv2.copyMakeBorder(character, 10, 10, pad, pad + 1, cv2.BORDER_CONSTANT)

        # Binarize and resize char
        ret, paddedChar = cv2.threshold(paddedChar, 127, 255, cv2.THRESH_BINARY)

        if Settings.showSteps["SegmentedChars"] == 1:
            cv2.imshow(("{}{}".format("ch", i)), paddedChar)

        chars.append(paddedChar)
        i = i + 1

    return chars
