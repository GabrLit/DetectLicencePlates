import math
import cv2

MIN_DIAG_SIZE_AWAY = 0.3
MAX_DIAG_SIZE_AWAY = 4.75

MAX_AREA_CHANGE = 1

MAX_WIDTH_CHANGE = 0.7
MAX_HEIGHT_CHANGE = 0.12

MAX_ANGLE = 9.0

MIN_NUMBER_OF_CHARS_IN_SET = 5

WIDTH_LP_SCALE_FACTOR = 1.03
HEIGHT_LP_SCALE_FACTOR = 1.3


def findCharSets(listOfChars):
    charSets = []

    # From list of all chars arrange every char with other chars into a set
    for char in listOfChars:
        oneCharSet = findMatchingChars(char, listOfChars)
        oneCharSet.append(char)

        if len(oneCharSet) < MIN_NUMBER_OF_CHARS_IN_SET:
            continue

        charSets.append(oneCharSet)

        # Create set without current set
        charSetsWithoutCurrent = list(set(listOfChars) - set(oneCharSet))

        # Recursive function
        rcsvcharSets = findCharSets(charSetsWithoutCurrent)
        for rcsvcharSet in rcsvcharSets:
            charSets.append(rcsvcharSet)

        break

    return charSets


def findMatchingChars(char, listOfChars):
    oneCharSet = []

    # Iterate big list and find complementary chars
    for matchingChar in listOfChars:

        # If matching char is the same as char continue.
        if matchingChar == char:
            continue

        # Check conditions

        # Distance between chars:
        Xdiff = abs(char.centerX - matchingChar.centerX)
        Ydiff = abs(char.centerY - matchingChar.centerY)
        distance = math.sqrt((Xdiff ** 2) + (Ydiff ** 2))

        # Angle between chars:
        if float(Xdiff) != 0.0:
            angleRad = math.atan(float(Ydiff) / float(Xdiff))
        else:
            angleRad = 0.0

        angleDeg = angleRad * (180.0 / math.pi)

        # Change in area
        areaChange = float(abs(matchingChar.area - char.area)) / float(char.area)

        # Change in width and height
        widthChange = float(abs(matchingChar.width - char.width)) / float(char.width)
        heightChange = float(abs(matchingChar.height - char.height)) / float(char.height)

        if ((char.diagonalSize * MAX_DIAG_SIZE_AWAY) > distance > (char.diagonalSize * MIN_DIAG_SIZE_AWAY) and
                angleDeg < MAX_ANGLE and
                areaChange < MAX_AREA_CHANGE and
                widthChange < MAX_WIDTH_CHANGE and
                heightChange < MAX_HEIGHT_CHANGE):
            oneCharSet.append(matchingChar)

    return oneCharSet


def extractLP(image, charSet):
    licencePlate = None

    charSet.sort(key=lambda char: char.centerX)

    # Calculate center of a licence plate
    LPCenterX = (charSet[0].centerX + charSet[len(charSet) - 1].centerX) / 2.0
    LPCenterY = (charSet[0].centerY + charSet[len(charSet) - 1].centerY) / 2.0

    LPCenter = LPCenterX, LPCenterY

    # Calculate width and height of licence plate
    LPWidth = int((charSet[len(charSet) - 1].x + charSet[len(charSet) - 1].width - charSet[0].x) * WIDTH_LP_SCALE_FACTOR)

    charHeightSum = 0
    for char in charSet:
        charHeightSum = charHeightSum + char.height

    LPHeight = int((charHeightSum / len(charSet)) * HEIGHT_LP_SCALE_FACTOR)

    # Correction in angle
    opposite = charSet[len(charSet) - 1].centerY - charSet[0].centerY

    # distance between first and last char
    Xdiff = abs(charSet[0].centerX - charSet[len(charSet) - 1].centerX)
    Ydiff = abs(charSet[0].centerY - charSet[len(charSet) - 1].centerY)
    distance = math.sqrt((Xdiff ** 2) + (Ydiff ** 2))

    angleRad = math.asin(opposite / distance)
    angleDeg = angleRad * (180.0 / math.pi)

    rotationMatrix = cv2.getRotationMatrix2D(tuple(LPCenter), angleDeg, 1.0)
    # Get original image parametrs
    height, width, channels = image.shape

    imageRotate = cv2.warpAffine(image, rotationMatrix, (width, height))
    imageCrop = cv2.getRectSubPix(imageRotate, (LPWidth, LPHeight), tuple(LPCenter))
    licencePlate = imageCrop

    return licencePlate
