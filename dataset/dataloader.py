from enum import StrEnum
import cv2

class CameraAngle(StrEnum):
    top_view = "top_view"
    horizontal_1 = "horizontal_1"
    horizontal_2 = "horizontal_2"
    horizontal_3 = "horizontal_3"

class VideoHandler:
    def __init__(self, path: str):
        self.path = path
        #self.video = cv2.VideoCapture(path)
        #self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.currentframe = 0

    def get_angle_stream(self, angle: CameraAngle):
        pass