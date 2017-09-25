# Imports
import glob
import cv2
import numpy as np
import pickle

chessboardSize = (9,6)
BLACK = (0,0,0)
GREEN = (0,255, 0)


def calibrateCamera(objpts, imgpts, size):
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size,None,None)
    if ret:
        return {'mtx': mtx, 'dist': dist}


def drawCorners(img, corners):
    for c in corners:
        center = tuple(c[0])
        cv2.circle(img, center, 4, GREEN, -1) # -1 means fill
    cv2.imshow('chessboard', img)
    cv2.waitKey(1)


def calibrate(path, save=True, ignore_old_calib=True):
    if not ignore_old_calib:
        lst = glob.glob('./*.p')
        if lst:
            return pickle.load(open('CalibrationParameters.p','rb'))
    objpts = []
    imgpts = []
    objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    files = glob.glob(path + '/*.png')
    size = None
    for imgFile in files:
        img = cv2.imread(imgFile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(imgFile)
        size = gray.shape
        print(size)
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            drawCorners(gray, corners)
            objpts.append(objp)
            imgpts.append(corners)
    result = calibrateCamera(objpts, imgpts, size)
    if save:
        pickle.dump(result, open('CalibrationParameters.p','wb'))
    return result

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pictures-dir", required=True, help="Directory with pictures used to calibrate")
    args = vars(ap.parse_args())
    result = calibrate(args['pictures_dir'], save=False)
    print(result['mtx'])
    print(result['dist'])
