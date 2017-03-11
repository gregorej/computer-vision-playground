# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2

event_loop_listeners = []


def start_event_loop():
    while True:
        key = cv2.waitKey(1) & 0xFF
        for listener in event_loop_listeners:
            listener(key)


def add_event_loop_listener(listener):
    event_loop_listeners.append(listener)


class Cropper(object):
    def __init__(self, window_name):
        self.__windowName = window_name
        self.__currentCrop = None
        self.__cropping = False
        self.__startPoint = None
        self.__onCrop = lambda x: x
        self.__image = None

    def on_crop(self, listener):
        self.__onCrop = listener

    def change_image(self, image):
        self.__image = image
        self.__reset()
        self.__refresh_image()

    def start(self, image):
        self.__image = image
        cv2.namedWindow(self.__windowName)
        cv2.setMouseCallback(self.__windowName, lambda event, x, y, req, param: self.__click_and_crop(event, x, y))
        add_event_loop_listener(lambda key: self.__key_listener(key))

    def __key_listener(self, key):
        self.__refresh_image()

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            self.__currentCrop = None
            self.__startPoint = None

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            if self.__currentCrop is not None:
                self.__onCrop(self.__currentCrop)
                self.__reset()

    def __click_and_crop(self, event, x, y):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__startPoint = (x, y)
            self.__cropping = True

        if self.__cropping:
            self.__currentCrop = (self.__startPoint, (x, y))

        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.__cropping = False
            self.__startPoint = None

    def __reset(self):
        self.__cropping = False
        self.__currentCrop = None
        self.__startPoint = None

    def __refresh_image(self):
        if self.__currentCrop is not None:
            cloned = self.__image.copy()
            cv2.rectangle(cloned, self.__currentCrop[0], self.__currentCrop[1], (0, 255, 0), 2)
            cv2.imshow(self.__windowName, cloned)
        else:
            cv2.imshow(self.__windowName, self.__image)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())


    def display_crop(crop):
        roi = image[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
        cv2.imshow("ROI", roi)

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args["image"])
    cropper = Cropper("image")
    cropper.on_crop(display_crop)
    cropper.start(image)

    start_event_loop()

    # close all open windows
    cv2.destroyAllWindows()
