from threading import Thread
import cv2

class VideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should be stopped
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
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

#Testing the VideoStream class
if __name__ == "__main__":
    vs = VideoStream(src=0).start()  # Use the default webcam
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            print("Failed to grab frame")
            break

        # Display the frame
        cv2.imshow("Frame", frame)

        # Stop the video stream when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()
