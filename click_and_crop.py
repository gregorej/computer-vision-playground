# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
regions = []
cropping = False
startPoint = None
currentCrop = None


def click_and_crop(event, x, y, req, param):
    # grab references to the global variables
    global cropping, startPoint, regions, currentCrop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        startPoint = (x, y)
        currentCrop = (startPoint, startPoint)
        cropping = True

    if cropping:
        currentCrop = (startPoint, (x, y))

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        regions.append(currentCrop)
        cropping = False
        startPoint = None

    refresh_image()


def refresh_image():
    global currentCrop, image
    if currentCrop is not None:
        cloned = image.copy()
        cv2.rectangle(cloned, currentCrop[0], currentCrop[1], (0, 255, 0), 2)
        cv2.imshow("image", cloned)
    else:
        cv2.imshow("image", image)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    refresh_image()
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if currentCrop is not None:
    roi = image[currentCrop[0][1]:currentCrop[1][1], currentCrop[0][0]:currentCrop[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
