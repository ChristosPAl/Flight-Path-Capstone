__author__ = "Hannes Hoettinger"

# import the necessary packages
from threading import Thread, Lock
import cv2
import os

# function call:
# vs = VideoStream(src=0).start()
# frame = vs.read()


class VideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        (self.grabbed, self.frame) = self.stream.read()
        #self.frame = cv2.flip(self.frame, -1)  # flip the first frame
        #self.frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.lock = Lock()

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()
            frame = cv2.flip(frame, -1)  # flip the frame
            # rotate the frame 90 degrees
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # acquire the lock, update the frame, and release the lock
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        # acquire the lock, copy the frame, and release the lock
        with self.lock:
            grabbed = self.grabbed
            frame = self.frame.copy()
        return grabbed, frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

        # added 2 lines of code to fix error: qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/capstone/Desktop/foobar/lib/python3.11/site-packages/cv2/qt/plugins"
        os.environ['QT_QPA_PLATFORM'] = 'xcb'


# tester main function 
if __name__ == "__main__":
    # Create an instance of the VideoCapture class
    video_capture = VideoStream(src=0)  # '/dev/video2')

    # Start capturing frames
    video_capture.start()

    while True:
        grabbed, frame = video_capture.read()
        if not grabbed:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Stop the video capture
    video_capture.stop()
    cv2.destroyAllWindows()