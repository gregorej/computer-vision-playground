# Imports
import glob
import cv2
import numpy as np
import pickle

chessboardSize = (9,6)
BLACK = (0,0,0)
GREEN = (0,255, 0)

def calibrateCamera(objpts , imgpts , size):
    ret,mtx,dist,_,_ =  cv2.calibrateCamera(objpts , imgpts, size,None,None)
    if (ret):
       dict = {}
       dict['mtx'] = mtx
       dict['dist'] = dist
       return dict

def drawCorners(img, corners):
    for c in corners:
        center = tuple(c[0])
        print center
        cv2.circle(img, center, 4, GREEN, -1) # -1 means fill
    cv2.imshow('chessboard',img)
    cv2.waitKey(1)

def calibrate(path ,size, save = True,ignore_old_calib = True):
    if(ignore_old_calib == False):
        lst = glob.glob('./*.p')
        if(lst):
            return  pickle.load(open('CalibrationParameters.p','rb'))
    objpts = []
    imgpts = []
    objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)
    files = glob.glob(path + '/*.jpg')
    for imgFile in files:
        img = cv2.imread(imgFile)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret , corners = cv2.findChessboardCorners(gray, chessboardSize,None)
        if(ret == True):
            drawCorners(gray, corners)
            objpts.append(objp)
            imgpts.append(corners)
    dict = calibrateCamera(objpts , imgpts , size)
    print dict
    if (save == True):
        pickle.dump(dict , open('CalibrationParameters.p','wb'))
    return dict

if __name__ == "__main__":
    calibrate('./calibration/dell',(540,960))
