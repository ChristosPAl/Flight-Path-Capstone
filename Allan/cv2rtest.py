import cv2
import time

# Attempt to open the video device
cap = cv2.VideoCapture('/dev/video2')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1 frame

start_time = time.time()
timeout = 10

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Check if the frame is valid
    if frame is None or frame.size == 0:
        print("Captured frame is invalid")
        break

    cv2.imshow('Imagetest', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()