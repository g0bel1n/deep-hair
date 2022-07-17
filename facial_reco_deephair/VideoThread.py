import time
import cv2
from threading import Thread


class VideoThread:
    def __init__(self, source: str):
        self.stream = cv2.VideoCapture(source)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

        self.fps = self.stream.get(
            cv2.CAP_PROP_FPS
        )  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"

        self.videoTime: int = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)) / self.fps

    def start(self):
        Thread(target=self.get).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                time.sleep(0.005)
                (self.ret, self.frame) = self.stream.read()
                self.videoTime = (
                    int(self.stream.get(cv2.CAP_PROP_POS_FRAMES)) / self.fps
                )

    def stop(self):
        self.stopped = True
