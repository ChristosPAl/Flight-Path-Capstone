__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
import time
import math
import pickle
from Classes import *
from MathFunctions import *
from DartsMapping import *
from Draw import *
from MyCalibration_1 import *

DEBUG = False

winName = "test2"

def cam2gray(cam):
    success, image = cam.read()
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return success, img_g

def getThreshold(cam, t):
    success, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 60, 255, 0)
    return thresh

def diff2blur(cam, t):
    _, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)
    ## kernel size important -> make accessible
    # filter noise from image distortions
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)
    return t_plus, blur

def getCorners(img_in):
    # number of features to track is a distinctive feature
    ## FeaturesToTrack important -> make accessible
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.08)  # k=0.08
    #print(edges)
    corners = np.int64(edges) #int0 before, threw error
    #print(corners)
    return corners

def filterCorners(corners):
    cornerdata = []
    tt = 0
    mean_corners = np.mean(corners, axis=0)
    for i in corners:
        xl, yl = i.ravel()
        # filter noise to only get dart arrow
        ## threshold important -> make accessible
        if abs(mean_corners[0][0] - xl) > 180:
            cornerdata.append(tt)
        if abs(mean_corners[0][1] - yl) > 120:
            cornerdata.append(tt)
        tt += 1
    corners_new = np.delete(corners, cornerdata, axis=0)  # delete corners to form new array
    return corners_new

def filterCornersLine(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x.item() * vy.item() / vx.item()) + y.item())
    righty = int(((cols - x.item()) * vy.item() / vx.item()) + y.item())
    cornerdata = []
    tt = 0
    for i in corners:
        xl, yl = i.ravel()
        # check distance to fitted line, only draw corners within certain range
        distance = dist(0, lefty, cols - 1, righty, xl, yl)
        if distance > 40:  ## threshold important -> make accessible
            cornerdata.append(tt)
        tt += 1
    corners_final = np.delete(corners, cornerdata, axis=0)  # delete corners to form new array
    return corners_final

def getRealLocation(corners_final, mount):
    if mount == "right":
        loc = np.argmax(corners_final, axis=0)
    else:
        loc = np.argmin(corners_final, axis=0)
    locationofdart = corners_final[loc]
    # check if dart location has neighbouring corners (if not -> continue)
    cornerdata = []
    tt = 0
    for i in corners_final:
        xl, yl = i.ravel()
        distance = abs(locationofdart.item(0) - xl) + abs(locationofdart.item(1) - yl)
        if distance < 40:  ## threshold important
            tt += 1
        else:
            cornerdata.append(tt)
    if tt < 3:
        corners_temp = cornerdata
        maxloc = np.argmax(corners_temp, axis=0)
        locationofdart = corners_temp[maxloc]
        print("### used different location due to noise!")
    return locationofdart

def getEllipseLineIntersection(Ellipse, M, lines_seg):
    center_ellipse = (Ellipse.x, Ellipse.y)
    circle_radius = Ellipse.a
    M_inv = np.linalg.inv(M)
    # find line circle intersection and use inverse transformation matrix to transform it back to the ellipse
    intersectp_s = []
    for lin in lines_seg:
        line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
        line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
        inter1, inter_p1, inter2, inter_p2 = intersectLineCircle(np.asarray(center_ellipse), circle_radius,
                                                                 np.asarray(line_p1), np.asarray(line_p2))
        if inter1:
            inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            if inter2:
                inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
                intersectp_s.append(inter_p1)
                intersectp_s.append(inter_p2)
    print(intersectp_s)
    return intersectp_s

def manipulateTransformationPoints(imCal, calData):
    print("manipulating transformation points")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('tx1', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty1', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx2', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty2', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx3', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty3', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx4', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty4', 'image', 0, 20, nothing)
    cv2.setTrackbarPos('tx1', 'image', 10)
    cv2.setTrackbarPos('ty1', 'image', 10)
    cv2.setTrackbarPos('tx2', 'image', 10)
    cv2.setTrackbarPos('ty2', 'image', 10)
    cv2.setTrackbarPos('tx3', 'image', 10)
    cv2.setTrackbarPos('ty3', 'image', 10)
    cv2.setTrackbarPos('tx4', 'image', 10)
    cv2.setTrackbarPos('ty4', 'image', 10)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    imCal_copy = imCal.copy()
    while True:
        cv2.imshow('image', imCal_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("got k, break")
            break
        # get current positions of four trackbars
        tx1 = cv2.getTrackbarPos('tx1', 'image') - 10
        ty1 = cv2.getTrackbarPos('ty1', 'image') - 10
        tx2 = cv2.getTrackbarPos('tx2', 'image') - 10
        ty2 = cv2.getTrackbarPos('ty2', 'image') - 10
        tx3 = cv2.getTrackbarPos('tx3', 'image') - 10
        ty3 = cv2.getTrackbarPos('ty3', 'image') - 10
        tx4 = cv2.getTrackbarPos('tx4', 'image') - 10
        ty4 = cv2.getTrackbarPos('ty4', 'image') - 10
        s = cv2.getTrackbarPos(switch, 'image')
        if s == 0:
            imCal_copy[:] = 0
        else:
            # transform the image to form a perfect circle
            transformation_matrix = transformation(imCal, calData, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4)
    return transformation_matrix

def getDarts(cam, calData, playerObj, GUI):
    finalScore = 0
    count = 0
    breaker = 0
    ## threshold important -> make accessible
    minThres = 2000 / 2
    maxThres = 15000 / 2
    # save score if score is below 1...
    old_score = playerObj.score
    # Read first image twice (issue somewhere) to start loop:
    _, _ = cam2gray(cam)
    _, _ = cam2gray(cam)
    # wait for camera
    time.sleep(0.1)
    success, t = cam2gray(cam)
    while success:
        # wait for camera
        time.sleep(0.1)
        # check if dart hit the board
        thresh = getThreshold(cam, t)
        print(cv2.countNonZero(thresh))
        ## threshold important
        if cv2.countNonZero(thresh) > minThres and cv2.countNonZero(thresh) < maxThres:
            # wait for camera vibrations
            time.sleep(0.2)
            # filter noise
            t_plus, blur = diff2blur(cam, t)
            # get corners
            corners = getCorners(blur)
            testimg = blur.copy()
            # dart outside?
            if corners.size < 40:
                print("### dart not detected")
                continue
            # filter corners
            corners_f = filterCorners(corners)
            # dart outside?
            if corners_f.size < 30:
                print("### dart not detected")
                continue
            # find left and rightmost corners#
            rows, cols = blur.shape[:2]
            corners_final = filterCornersLine(corners_f, rows, cols)
            print(f"corners_final: {corners_final}")
            _, thresh = cv2.threshold(blur, 60, 255, 0)
            # check if it was really a dart
            print(cv2.countNonZero(thresh))
            if cv2.countNonZero(thresh) > maxThres * 2:
                continue
            print("Dart detected")
            # dart was found -> increase counter
            breaker += 1
            dartInfo = DartDef()
            # get final darts location
            try:
                dartInfo.corners = corners_final.size
                print("hello")
                locationofdart = getRealLocation(corners_final, "right")
                print(f"location of dart: {locationofdart}")
                # check for the location of the dart with the calibration
                # dartloc = getTransformedLocation(locationofdart.item(0), locationofdart.item(1), calData)
                print(f"locationofdart: {locationofdart[0][0][0][0]}, {locationofdart[0][0][0][1]}")
                print(f"calData: {calData.transformation_matrix}")
                dartloc = getTransformedLocation(locationofdart[0][0][0][0], locationofdart[0][0][0][1], calData)
                print(f"dartloc: {dartloc}")
                # detect region and score
                dartInfo = getDartRegion(dartloc, calData)
                cv2.circle(testimg, (locationofdart.item(0), locationofdart.item(1)), 10, (255, 255, 255), 2, 8)
                cv2.circle(testimg, (locationofdart.item(0), locationofdart.item(1)), 2, (0, 255, 0), 2, 8)
            except:
                print("1Something went wrong in finding the darts location!")
                breaker -= 1
                continue
            print(dartInfo.base, dartInfo.multiplier)
            if breaker == 1:
                GUI.dart1entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart1entry.get())
                cv2.imwrite("frame2.jpg", testimg)  # save dart1 frame
            elif breaker == 2:
                GUI.dart2entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart2entry.get())
                cv2.imwrite("frame3.jpg", testimg)  # save dart2 frame
            elif breaker == 3:
                GUI.dart3entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart3entry.get())
                cv2.imwrite("frame4.jpg", testimg)  # save dart3 frame
            playerObj.score -= dart
            if playerObj.score == 0 and dartInfo.multiplier == 2:
                playerObj.score = 0
                breaker = 3
            elif playerObj.score <= 1:
                playerObj.score = old_score
                breaker = 3
            # save new diff img for next dart
            t = t_plus
            if playerObj.player == 1:
                GUI.e1.delete(0, 'end')
                GUI.e1.insert(10, playerObj.score)
            else:
                GUI.e2.delete(0, 'end')
                GUI.e2.insert(10, playerObj.score)
            finalScore += (dartInfo.base * dartInfo.multiplier)
            if breaker == 3:
                break
        # missed dart
        elif cv2.countNonZero(thresh) < maxThres / 2:
            continue
        # if player enters zone - break loop
        elif cv2.countNonZero(thresh) > maxThres / 2:
            break
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break
        count += 1
    GUI.finalentry.delete(0, 'end')
    GUI.finalentry.insert(10, finalScore)
    print(f"final score: {finalScore}")

if __name__ == '__main__':
    getDarts()