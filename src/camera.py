import numpy as np
from vmbpy import VmbSystem, PixelFormat, VmbCameraError, VmbFeatureError
from sys import exit

# Wrapper for vmbpy
class Camera:

    def __init__(self):
        self._vmb = VmbSystem.get_instance()

    def __enter__(self):
        self._vmb.__enter__()
        cams = self._vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            exit(1)
        self.cam = cams[0]
        try:
            self.cam.__enter__()
        except VmbCameraError as e:
            if "Read" in str(e) or "invalid Mode" in str(e):
                print("Camera locked — close Vimba Viewer and retry.")
                self._vmb.__exit__(None, None, None)
                exit(1)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cam.__exit__(exc_type, exc_val, exc_tb)
        self._vmb.__exit__(exc_type, exc_val, exc_tb)

    def print_info(self):
        print(f"Camera   : {self.cam.get_name()}  ({self.cam.get_id()})")
        print(f"Model    : {self.cam.get_model()}")
        print(f"Interface: {self.cam.get_interface_id()}")
        print("─" * 44)

    def setup(self, fmt=PixelFormat.Mono8,
              exposure_us: float = 1500.0,
              gain: float = 0.0):
        for attr, val in [("ExposureTime", exposure_us), ("Gain", gain)]:
            try:
                getattr(self.cam, attr).set(val)
            except (AttributeError, VmbFeatureError):
                pass
        try:
            self.cam.set_pixel_format(fmt)
        except (AttributeError, VmbFeatureError):
            pass

    def grab(self, timeout_ms: int = 2000) -> np.ndarray:
        return self.cam.get_frame(timeout_ms=timeout_ms).as_numpy_ndarray()
