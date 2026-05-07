import cv2
import numpy as np

# Wrapper for CV2 to easily use a window.
class CVWindow:

    def __init__(self, monitor, name: str, fullscreen: bool = True):
        self.w, self.h = monitor.width, monitor.height
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, monitor.x, monitor.y)
        prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, prop)

    def show(self, img: np.ndarray):
        cv2.imshow(self.name, img)
