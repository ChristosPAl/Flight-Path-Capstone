from threading import Event
import sys
import math
import pickle
import os.path
from im2figure import *
from numpy.linalg import inv
from MathFunctions import *
from Classes import *
from Draw import *
from VideoCapture import VideoStream
import numpy as np
import cv2

def ellipse2circle(Ellipse):
    angle = (Ellipse.angle) * math.pi / 180
    x = Ellipse.x
    y = Ellipse.y
    a = Ellipse.a
    b = Ellipse.b

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    R1 = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    return M

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

    print((intersectp_s))

    return intersectp_s

def findEllipse(thresh2, image_proc_img):
    Ellipse = EllipseDef()
    print("found ellipse")
    contours, hierarchy = cv2.findContours(thresh2, 1, 2)

    minThresE = 200000 / 4
    maxThresE = 1000000 / 4
    print(f"number of contours = {len(contours)}")
    ## contourArea threshold important -> make accessible
    for cnt in contours:
        try:  # threshold critical, change on demand?
            if minThresE < cv2.contourArea(cnt) < maxThresE:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(image_proc_img, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]
                print(ellipse[1])
                center_ellipse = (x, y)

                a = a / 2
                b = b / 2

                cv2.ellipse(image_proc_img, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0,
                            (255, 0, 0))

                # Calculate the endpoints of the major axis
                angle_rad = np.deg2rad(angle)
                x1 = int(x + a * np.cos(angle_rad))
                y1 = int(y + a * np.sin(angle_rad))
                x2 = int(x - a * np.cos(angle_rad))
                y2 = int(y - a * np.sin(angle_rad))

                # Draw the major axis
                cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                print("no ellipse of correct size")
        # corrupted file
        except:
            print("error")
            return Ellipse, image_proc_img

    Ellipse.a = a
    Ellipse.b = b
    Ellipse.x = x
    Ellipse.y = y
    Ellipse.angle = angle
    return Ellipse, image_proc_img

def findSectorLines(edged, image_proc_img, angleZone1, angleZone2):
    p = []
    intersectp = []
    lines_seg = []
    counter = 0
    sector_line_count = 0  # Counter for sector lines

    # fit line to find intersec point for dartboard center point
    lines = cv2.HoughLines(edged, 1, np.pi / 80, 100, 100)
    print(f"length of lines: {len(lines)}")

    for line in lines:
        ## sector angles important -> make accessible
        for rho, theta in line:
            print(f"rho: {rho}, theta: {theta}")
            # split between horizontal and vertical lines (take only lines in certain range)
            print(f"{theta} > {np.pi / 180 * angleZone1[0]} and {theta} < {np.pi / 180 * angleZone1[1]}")

            if theta > (np.pi / 180) * angleZone1[0] and theta < (np.pi / 180) * angleZone1[1]:
                print("entered if statement")

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))

                for line in lines:
                    for rho1, theta1 in line:
                        print(f"{theta1} > {np.pi / 180 * angleZone2[0]} and {theta1} < {np.pi / 180 * angleZone2[1]}")

                        if theta1 > (np.pi / 180) * angleZone2[0] and theta1 < (np.pi / 180) * angleZone2[1]:
                            print("entered second if statement")
                            a = np.cos(theta1)
                            b = np.sin(theta1)
                            x0 = a * rho1
                            y0 = b * rho1
                            x3 = int(x0 + 2000 * (-b))
                            y3 = int(y0 + 2000 * (a))
                            x4 = int(x0 - 2000 * (-b))
                            y4 = int(y0 - 2000 * (a))

                            if y1 == y2 and y3 == y4:  # Horizontal Lines
                                diff = abs(y1 - y3)
                            elif x1 == x2 and x3 == x4:  # Vertical Lines
                                diff = abs(x1 - x3)
                            else:
                                diff = 0

                            print(diff)

                            if diff < 200 and diff != 0:
                                continue

                            # Display lines
                            cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                            p.append((x1, y1))
                            p.append((x2, y2))
                            p.append((x3, y3))
                            p.append((x4, y4))
                            print(p)

                            intersectpx, intersectpy = intersectLines(p[counter], p[counter + 1], p[counter + 2], p[counter + 3])

                            # Consider only intersection close to the center of the image
                            if intersectpx < 200 or intersectpx > 900 or intersectpy < 200 or intersectpy > 900:
                                continue

                            intersectp.append((intersectpx, intersectpy))

                            lines_seg.append([(x1, y1), (x2, y2)])
                            lines_seg.append([(x3, y3), (x4, y4)])

                            cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                            # Increment the sector line counter
                            sector_line_count += 1

                            # Point offset
                            counter = counter + 4

    print(f"Number of sector lines found: {sector_line_count}")
    return lines_seg, image_proc_img

def test_function(image_path):
    # Read the image from the file
    image_proc_img = cv2.imread(image_path)
    if image_proc_img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image_proc_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological closing
    kernel = np.ones((5, 5), np.uint8)
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("thresh2", thresh2)
    cv2.waitKey(0)
    # Call the findEllipse function
    Ellipse, image_proc_img = findEllipse(thresh2, image_proc_img)
    print("Ellipse properties:")
    print(f"a: {Ellipse.a}, b: {Ellipse.b}, x: {Ellipse.x}, y: {Ellipse.y}, angle: {Ellipse.angle}")

    # Test finding sector lines
    edged = cv2.Canny(thresh2, 50, 150)
    cv2.imshow("edge detection", edged)
    cv2.waitKey(0)
    if Ellipse.angle is not None:
        if "right" == "right":  # Replace with actual mount condition if needed
            print("mount is right, ellipse angle:")
            print(Ellipse.angle)
            angleZone1 = (Ellipse.angle - 5, Ellipse.angle + 5)  # Adjust these values as needed
            angleZone2 = (Ellipse.angle - 100, Ellipse.angle - 80)  # Adjust these values as needed
            lines_seg, image_proc_img = findSectorLines(edged, image_proc_img, angleZone1, angleZone2)
    M = ellipse2circle(Ellipse)

    intersectp_s = getEllipseLineIntersection(Ellipse, M, lines_seg)

    source_points = intersectp_s

    # Display the result
    cv2.circle(image_proc_img, (int(source_points[0][0]), int(source_points[0][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[1][0]), int(source_points[1][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[2][0]), int(source_points[2][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[3][0]), int(source_points[3][1])), 3, (255, 0, 0), 2, 8)

    calData = CalibrationData()

    

    winName2 = "th circles?"
    cv2.namedWindow(winName2, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winName2, image_proc_img)
    cv2.waitKey(0)

    end = cv2.waitKey(0)
    if end == 13:
        cv2.destroyAllWindows()
        #return source_points

if __name__ == '__main__':
    print("Testing findEllipse function")
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, 'frame1.jpg')
    test_function(image_path)
