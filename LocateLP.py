import random
import cv2
import numpy as np
import math
import Settings
import MatchChars

MIN_AREA = 60
MIN_WIDTH = 4
MIN_HEIGHT = 12
MIN_RATIO = 1.25
MAX_RATIO = 8
MIN_SOLIDITY = 0.10


class CharProperties:

    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)
        [self.x, self.y, self.width, self.height] = self.boundingRect

        self.area = self.width * self.height

        self.centerX = (self.x + self.x + self.width) / 2
        self.centerY = (self.y + self.y + self.height) / 2

        self.diagonalSize = math.sqrt((self.width ** 2) + (self.height ** 2))
        self.ratio = float(self.width) / float(self.height)


def init(image):
    empty = np.zeros_like(image)
    empty2 = np.zeros_like(image)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 5, 30, 30)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    if Settings.showSteps["GrayScale"] == 1:
        cv2.imshow("GrayScale", gray)
        cv2.waitKey(0)

    if Settings.showSteps["Binary"] == 1:
        cv2.imshow("Binary", binary)
        cv2.waitKey(0)

    # Finding contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Early candidates for characters
    early_candidates = []
    for c in contours:
        boundingRect = cv2.boundingRect(c)
        [x, y, width, height] = boundingRect
        area = width * height
        ratio = float(height) / float(width)
        solidity = cv2.contourArea(c) / float(width * height)

        # Checking area of character, minimal width and height, minimal width to height ratio, solidity.
        if area > MIN_AREA and width > MIN_WIDTH and height > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO and solidity > MIN_SOLIDITY:
            early_candidates.append(c)

    if Settings.showSteps["EarlyCandidates"] == 1:
        if len(early_candidates) == 0:
            print("No early candidates found")
        else:
            print("Found: {} early candidates".format(len(early_candidates)))
            cv2.drawContours(empty, early_candidates, -1, (0, 255, 0))
            cv2.imshow("EarlyCandidates", empty)
            cv2.waitKey(0)

    # Give early candidates properties
    listOfChars = []
    for c in early_candidates:
        listOfChars.append(CharProperties(c))

    # Find all matching sets of chars in image
    charSets = MatchChars.findCharSets(listOfChars)

    if Settings.showSteps["CharSets"] == 1:
        print("Found: {} matching character sets".format(len(charSets)))

        for charSet in charSets:
            randomB = random.randint(0, 255)
            randomG = random.randint(0, 255)
            randomR = random.randint(0, 255)

            contours = []

            for char in charSet:
                contours.append(char.contour)

            cv2.drawContours(empty2, contours, -1, (randomB, randomG, randomR))
        cv2.imshow("CharSets", empty2)
        cv2.waitKey(0)

    licencePlates = []
    for charSet in charSets:  # for each group of matching chars
        licencePlate = MatchChars.extractLP(image, charSet)  # attempt to extract plate

        if licencePlate is not None:
            licencePlates.append(licencePlate)  # add to list of possible plates


    if Settings.showSteps["LPCandidates"] == 1:
        print("Found: {} candidates for licence plates".format(len(licencePlates)))
        for licencePlate in licencePlates:
            cv2.imshow("licencePlate", licencePlate)
            cv2.waitKey(0)

    return licencePlates
