import cv2
import numpy as np


def resize_image(img, target_size):
    return cv2.resize(img, (target_size[1], target_size[0]))


def normalize_histogram(image):
    norm = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    norm[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    norm[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    norm[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    return norm
