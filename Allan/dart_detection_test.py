import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pickle

def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def getRealLocation(corners_final, mount, image):
    if mount == "right":
        loc = np.argmax(corners_final[:, 0])  # Get the index of the maximum x-coordinate
    else:
        loc = np.argmin(corners_final[:, 0])  # Get the index of the minimum x-coordinate
    locationofdart = corners_final[loc]

    # check if dart location has neighbouring corners (if not -> continue)
    cornerdata = []
    tt = 0
    for i in corners_final:
        xl, yl = i.ravel()
        distance = abs(locationofdart[0] - xl) + abs(locationofdart[1] - yl)
        if distance < 40:  ## threshold important. initial value 40
            tt += 1
        else:
            cornerdata.append(tt)
    if tt < 3:
        corners_temp = corners_final[cornerdata]
        maxloc = np.argmax(corners_temp[:, 0], axis=0)
        locationofdart = corners_temp[maxloc]
        print("### used different location due to noise!")

    # Draw a circle at the dart's location
    xl, yl = locationofdart
    cv2.circle(image, (xl, yl), 10, (0, 0, 255), -1)  # Red circle
    cv2.putText(image, 'Dart', (xl + 15, yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image with the dart location
    cv2.imshow('Dart Location', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return locationofdart

def getThreshold(image, ref_img):
    #success, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(ref_img, image)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 60, 255, 0)
    return thresh

def diff2blur(image, ref_img):
    #_, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(ref_img, image)
    ## kernel size important -> make accessible
    # filter noise from image distortions
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)
    return image, blur

def filterCorners(corners, img):
    cornerdata = []
    tt = 0
    mean_corners = np.mean(corners, axis=0)
    print(f"Mean corners: {mean_corners}")
    for i in corners:
        xl, yl = i.ravel()
        # filter noise to only get dart arrow
        ## threshold important -> make accessible
        if abs(mean_corners[0][0] - xl) > 180: #originally set to 180
            cornerdata.append(tt)
        if abs(mean_corners[0][1] - yl) > 120: #originally set to 120
            cornerdata.append(tt)
        tt += 1
    corners_new = np.delete(corners, cornerdata, axis=0)  # delete corners to form new array
    
    # Display filtered corners
    for i in corners_new:
        xl, yl = i.ravel()
        cv2.circle(img, (xl, yl), 3, (0, 255, 0), -1)  # Green circles for filtered corners
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Corners")
    plt.show()
    
    return corners_new

def filterCornersLine(corners, rows, cols, img):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x.item() * vy.item() / vx.item()) + y.item())
    righty = int(((cols - x.item()) * vy.item() / vx.item()) + y.item())

    print(f"lefty value is this {lefty}, righty value is {righty}")
    
    cornerdata = []
    tt = 0
    for i in corners:
        xl, yl = i.ravel()
        # check distance to fitted line, only draw corners within certain range
        distance = dist(0, lefty, cols - 1, righty, xl, yl)
        if distance > 20:  ## threshold important -> make accessible, intiali value 40
            cornerdata.append(tt)
        tt += 1
    corners_final = np.delete(corners, cornerdata, axis=0)  # delete corners to form new array
    
    # # Print the final corners
    # print("Filtered Corners:")
    # for corner in corners_final:
    #     print(corner.ravel())
    
    # Visualization
    for i in corners:
        xl, yl = i.ravel()
        cv2.circle(img, (xl, yl), 3, (255, 0, 0), -1)  # Red circles for original corners
    for i in corners_final:
        xl, yl = i.ravel()
        cv2.circle(img, (xl, yl), 3, (0, 255, 0), -1)  # Green circles for filtered corners
    cv2.line(img, (0, lefty), (cols - 1, righty), (0, 255, 255), 2)  # Yellow line for fitted line
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Corners and Fitted Line")
    plt.show()
    
    return corners_final

def detectAndDisplayCorners(img):
    # Shi-Tomasi corner detection function 
    # We are detecting only 100 best corners here 
    # You can change the number to get desired result. 
    corners = cv2.goodFeaturesToTrack(img, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.08)
  
    # Convert corners values to integer 
    # So that we will be able to draw circles on them 
    corners = np.int64(corners) 
  
    # Draw red color circles on all corners 
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1) 

    # Display detected corners
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Corners")
    plt.show()
    
    return corners

# Example usage in main function
if __name__ == "__main__":
    # # Construct the absolute path to the image file
    # script_dir = os.path.dirname(__file__)
    # image_path = os.path.join(script_dir, 'dart_in_board.jpg')

    # # Verify the constructed path
    # print(f"Attempting to read image from: {image_path}")

    # # Open the image file
    # img = cv2.imread(image_path)

    # if img is None:
    #     print(f"Cannot read image from {image_path}")
    #     exit()
    script_dir = os.path.dirname(__file__)
    image_path1 = os.path.join(script_dir, 'dart_in_board.jpg')
    image_path2 = os.path.join(script_dir, 'no_dart_in_board.jpg')
    # Open the image file
    image1_wDart= cv2.imread(image_path1)
    image2_noDart = cv2.imread(image_path2)

    if image1_wDart is None:
        print("Cannot read image")
        exit()
    
    # Convert frame to grayscale
    gray1_wDart = cv2.cvtColor(image1_wDart, cv2.COLOR_BGR2GRAY)
    gray2_noDart = cv2.cvtColor(image2_noDart, cv2.COLOR_BGR2GRAY)
    thresh = getThreshold(gray1_wDart,gray2_noDart)
    blur1, blur2 = diff2blur(gray1_wDart, gray2_noDart)
    cv2.imshow('Blurred Difference Image', blur2)

    #img = blur2
    # Detect and display corners
    corners = detectAndDisplayCorners(blur2)
    

    # Convert image to grayscale 
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # # Apply GaussianBlur to the frame
    # blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # # Apply threshold to the frame
    # _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY) 

    # Filter and display corners
    filtered_corners = filterCorners(corners, image1_wDart)

    # # Further filter corners using line fitting
    rows, cols = blur2.shape[:2]
    final_corners = filterCornersLine(filtered_corners, rows, cols, image1_wDart)

    # locationofdart = getRealLocation(final_corners, "right", image1_wDart)
    # print(f"Dart location: {locationofdart}")

    # De-allocate any associated memory usage   
    if cv2.waitKey(0) & 0xff == 27:  
        cv2.destroyAllWindows()
