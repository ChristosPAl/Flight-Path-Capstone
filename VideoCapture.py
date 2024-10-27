# __author__ = "Hannes Hoettinger"

# # import the necessary packages
# from threading import Thread
# from picamera2 import Picamera2
# import numpy as np
# import cv2

# # function call:
# # vs = VideoStream().start()
# # frame = vs.read()


# class VideoStream:
#     def __init__(self, src=0):
#         # initialize the video camera stream and read the first frame
#         # from the stream
#         self.camera = Picamera2()
#         self.camera.configure(self.camera.create_preview_configuration(main={"size": (800, 600)}))
#         self.camera.start()
#         self.frame = self.camera.capture_array()
#         self.src = src

#         # initialize the variable used to indicate if the thread should
#         # be stopped
#         self.stopped = False
#         print("VideoStream initialized")

#     def start(self):
#         # start the thread to read frames from the video stream
#         Thread(target=self.update, args=()).start()
#         print("VideoStream thread started")
#         return self

#     def update(self):
#         # keep looping infinitely until the thread is stopped
#         while True:
#             # if the thread indicator variable is set, stop the thread
#             if self.stopped:
#                 print("VideoStream thread stopping")
#                 return

#             # otherwise, read the next frame from the stream
#             self.frame = self.camera.capture_array()
#             print("Frame captured")

#     def read(self):
#         # return the frame most recently read
#         return True, self.frame

#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True
#         self.camera.stop()
#         print("VideoStream stopped")


# if __name__ == "__main__":
#     vs = VideoStream().start()
#     try:
#         while True:
#             grabbed, frame = vs.read()
#             if not grabbed:
#                 print("Frame not grabbed")
#                 break

#             # Display the frame
#             cv2.imshow("Frame", frame)
#             print("Displaying frame")

#             # Break the loop on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Exiting loop")
#                 break
#     except KeyboardInterrupt:
#         print("Keyboard interrupt received")
#         pass
#     finally:
#         vs.stop()
#         cv2.destroyAllWindows()
#         print("Video stream stopped and windows destroyed")




__author__ = "Hannes Hoettinger"

# import the necessary packages
from threading import Thread
import cv2

# function call:
# vs = VideoStream(src=0).start()
# frame = vs.read()


class VideoStream:
    def __init__(self, src=2):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

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
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

        #added 2 lines of code to fix error: qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/capstone/Desktop/foobar/lib/python3.11/site-packages/cv2/qt/plugins"
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'


#tester main function 
if __name__ == "__main__":
    # Create an instance of the VideoCapture class
    video_capture = VideoStream(src='/dev/video2')

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