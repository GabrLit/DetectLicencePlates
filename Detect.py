import cv2
import argparse
import LocateLP
import SegmentLP
import Settings
import Recognise
import Authorize
import time
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def init(image):
    start_time = time.time()
    # Initialize settings
    Settings.init()
    WIDTH = 800

    # Scale image
    # Calculate proportions
    r = WIDTH / image.shape[1]
    dim = (WIDTH, int(image.shape[0] * r))

    # Resize
    image = cv2.resize(image, dim)

    # Show scaled original image
    if Settings.showSteps["Original"] == 1:
        cv2.imshow("Original", image)
        cv2.waitKey(0)

    # Locate licence plate in image
    licencePlates = LocateLP.init(image)

    for licencePlate in licencePlates:
        chars = SegmentLP.init(licencePlate)
        if chars != 0:
            string = Recognise.init(chars)
            print("\nDetected licence plate numbers: \n{}".format(string))
            authorized = Authorize.init(string)
            if authorized:
                print("\nCar with licence plate: {} is allowed to enter".format(string))
            else:
                print("\nCar with licence plate: {} is NOT allowed to enter".format(string))

    print("\nExecution time:\n --- %s seconds ---" % (time.time() - start_time))
    cv2.waitKey(0)
    return authorized


# TEMPORARY READ IMAGE
ap = argparse.ArgumentParser()
ap.add_argument("--input", "-i", required=True)
args = vars(ap.parse_args())

image = cv2.imread(args['input'])

init(image)
