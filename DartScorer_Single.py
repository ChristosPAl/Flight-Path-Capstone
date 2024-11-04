import cv2
import time
import numpy as np
from threading import Thread
from Classes import *
from Draw import *
from MathFunctions import *
from VideoCapture import VideoStream

DEBUG = False
winName = "Dart Detection"

def getDarts(cam, calData, playerObj, GUI):
    finalScore = 0
    count = 0
    breaker = 0
    minThres = 1000  # Adjusted threshold for single camera
    maxThres = 5000

    old_score = playerObj.score

    # Read first image twice (issue workaround) to start loop:
    _, _ = cam.read()
    time.sleep(0.1)
    success, t_img = cam2gray(cam)  # First frame
    while success:
        time.sleep(0.1)  # Wait for stability
        # Check if dart hit the board by comparing current and reference frames
        thresh = getThreshold(cam, t_img)
        print(cv2.countNonZero(thresh))

        # Dart detection logic
        if minThres < cv2.countNonZero(thresh) < maxThres:
            time.sleep(0.2)  # Wait for vibrations to settle
            _, blur = diff2blur(cam, t_img)  # Filter noise

            # Get corners
            corners = getCorners(blur)
            if corners.size < 40:
                print("### Dart not detected")
                continue

            # Filter corners to find the dart location
            filtered_corners = filterCorners(corners)
            rows, cols = blur.shape[:2]
            corners_final = filterCornersLine(filtered_corners, rows, cols)

            # Check if it was really a dart
            if cv2.countNonZero(thresh) > maxThres * 2:
                continue

            print("Dart detected")
            breaker += 1
            dartInfo = DartDef()

            try:
                locationofdart = getRealLocation(corners_final, "center")  # Assuming central mount
                dartloc = getTransformedLocation(locationofdart.item(0), locationofdart.item(1), calData)

                # Detect the dart’s score region on the dartboard
                dartInfo = getDartRegion(dartloc, calData)
                cv2.circle(blur, (locationofdart.item(0), locationofdart.item(1)), 10, (255, 255, 255), 2, 8)
                cv2.circle(blur, (locationofdart.item(0), locationofdart.item(1)), 2, (0, 255, 0), 2, 8)
            except:
                print("Error finding dart location!")
                breaker -= 1
                continue

            # Update GUI entries with dart scores
            if breaker == 1:
                GUI.dart1entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart1entry.get())
            elif breaker == 2:
                GUI.dart2entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart2entry.get())
            elif breaker == 3:
                GUI.dart3entry.insert(10, str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart3entry.get())
            
            # Update player score
            playerObj.score -= dart
            if playerObj.score == 0 and dartInfo.multiplier == 2:
                playerObj.score = 0
                breaker = 3
            elif playerObj.score <= 1:
                playerObj.score = old_score
                breaker = 3

            # Prepare for next dart throw
            t_img = blur  # Save this blurred image as reference for next dart
            if playerObj.player == 1:
                GUI.e1.delete(0, 'end')
                GUI.e1.insert(10, playerObj.score)
            else:
                GUI.e2.delete(0, 'end')
                GUI.e2.insert(10, playerObj.score)
            
            finalScore += (dartInfo.base * dartInfo.multiplier)
            if breaker == 3:
                break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break
        count += 1

    GUI.finalentry.delete(0, 'end')
    GUI.finalentry.insert(10, finalScore)
    print("Final score:", finalScore)

# Initialize and start the dart detection with a single camera
if __name__ == '__main__':
    print("Starting dart detection with single camera")

    # Set up single camera
    cam = VideoStream(src=0).start()  # Assuming the camera is at index 0
    time.sleep(1.0)  # Allow camera to warm up

    # Dummy initialization for testing
    calData = CalibrationData()  # Load or define calibration data
    playerObj = Player()
    GUI = GUIDef()  # Assuming GUIDef is predefined in the project

    getDarts(cam, calData, playerObj, GUI)
    cam.stop()
