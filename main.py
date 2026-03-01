import cv2
import numpy as np
import screeninfo
from screeninfo.common import Monitor as Monitor
from numba import njit, prange, int16, uint8, float32
from vmbpy import *
from sys import exit

class Display:
    def __init__(self, monitor, name : str):
        self.width, self.height = monitor.width, monitor.height
        self.window_name = name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

class Camera:
    def __init__(self):
        self._vmb = VmbSystem.get_instance()

    def __enter__(self):
        self._vmb.__enter__()
        cams = self._vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            exit()
        self.cam = cams[0]
        try:
            self.cam.__enter__()
        except VmbCameraError as e:
            if AccessMode.Read in str(e) or "invalid Mode" in str(e):
                print("Camera is locked in read-only mode.")
                print("Please close Vimba Viewer or any other application using the camera and try again.")
                self._vmb.__exit__(None, None, None)
                exit()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cam.__exit__(exc_type, exc_val, exc_tb)
        self._vmb.__exit__(exc_type, exc_val, exc_tb)

    def print_camera_info(self):
        print("Camera ID:", self.cam.get_id())
        print("Model:", self.cam.get_model())
        print("Name:", self.cam.get_name())
        print("Interface ID:", self.cam.get_interface_id())
        print("-" * 40)

    def setup_camera(self, format, exposure_time: float, gain: float):
        try:
            self.cam.set_pixel_format(format)
        except (AttributeError, VmbFeatureError):
            pass

        try:
            self.cam.ExposureTime.set(exposure_time)
        except (AttributeError, VmbFeatureError):
            pass

        try:
            self.cam.Gain.set(gain)
        except (AttributeError, VmbFeatureError):
            pass

    def get_frame(self, timeout_ms=2000):
        return self.cam.get_frame(timeout_ms=timeout_ms)

class Optimizer:
    def __init__(self, cam):
        self.monitors = screeninfo.get_monitors()

        # Sets up three display windows, one representing the SLM on the main
        # computer screen, one for the sensor, and one for the SLM itself
        self.display = Display(self.monitors[0], "Display_Control")
        cv2.setWindowProperty(self.display.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        self.sensor = Display(self.monitors[0], "Sensor_Data")
        cv2.setWindowProperty(self.sensor.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if len(self.monitors) < 2:
            print("SLM not detected. Exiting.")
            exit()
        else:
            print("SLM detected.")
            self.slm = Display(self.monitors[1], "SLM_Control")
            if self.slm.width > self.display.width or self.slm.height > self.display.height:
                raise ValueError("SLM can not have a larger display than the computer display for viewing purposes.")

        # The starting image estimation
        self.image = np.ones((self.display.height, self.display.width), dtype = np.float32) * 128
        self.image = np.ascontiguousarray(self.image)

        self.update_control_image(self.image)

        self.cam = cam

        self.delta_u_magnitude = 5

        # These hyperparameters have been approximated, check the disclaimer in the README
        self.beta_1 = 0.7
        self.beta_2 = 0.95
        self.alpha = 0.07

        self.m = np.zeros_like(self.image)
        self.v = np.zeros_like(self.image)
        self.V_hat = np.ones_like(self.image) * -np.inf

        self.k = 0

        self.epsilon = np.finfo(np.float32).eps

    def iteration(self):
        m_previous = self.m
        v_previous = self.v
        V_hat_previous = self.V_hat

        perturbation = self.get_perturbation()
        if self.image.shape != perturbation.shape:
            raise ValueError("Image and Perturbation sizes do not match")

        # Calculates the positive performance index
        perturbed_image = np.ascontiguousarray(self.image)
        Optimizer.apply_perturbation(perturbed_image, perturbation)
        self.update_control_image(perturbed_image)
        I = self.cam.get_frame().as_numpy_ndarray()
        J_plus = Optimizer.compute_performance_index(I)

        # Calculates the positive performance index
        perturbed_image = np.ascontiguousarray(self.image)
        Optimizer.apply_perturbation(perturbed_image, -perturbation)
        self.update_control_image(perturbed_image)
        I = self.cam.get_frame().as_numpy_ndarray()
        J_minus = Optimizer.compute_performance_index(I)

        delta_J = J_plus - J_minus
        g = delta_J

        # Calculate the first and second momentum terms
        self.m = self.beta_1 * m_previous + (1 - self.beta_1) * g * perturbation
        self.v = self.beta_2 * v_previous + (1 - self.beta_2) * (g ** 2) * (perturbation ** 2)

        # Corrected momentum terms
        m_hat = self.m / (1 - (self.beta_1 ** self.k) + self.epsilon)
        v_hat = self.v / (1 - (self.beta_2 ** self.k) + self.epsilon)

        self.V_hat = np.maximum(V_hat_previous, v_hat).max()

        # Calculate the new control voltage image
        delta_image = ((self.alpha * m_hat) / (self.V_hat + self.epsilon)) * (perturbation ** 2)
        self.image += delta_image
        print(f"={self.image}")

        self.update_control_image(self.image)

        self.k += 1

    def get_perturbation(self) -> np.ndarray:
        bernoulli_distribution = np.random.binomial(
            n = 1, # since it is a bernoulli distribution
            p = 0.5,
            size = (self.display.height, self.display.width)
            )
        return (2 * bernoulli_distribution - 1) *self.delta_u_magnitude

    # Currently unsure if this njit approach is the fastest
    @njit(parallel=True, fastmath=True, cache=True)
    def apply_perturbation(image: np.ndarray, perturbation: np.ndarray) -> None:
        rows, cols = image.shape
        for r in prange(rows):
            for c in range(cols):
                val = int16(image[r, c]) + perturbation[r, c]

                if val > 255:
                    val = 255
                elif val < 0:
                    val = 0

                image[r, c] = uint8(val)

    # Currently unsure if this njit approach is the fastest
    @njit(parallel=True, fastmath=True, cache=True)
    def compute_performance_index(I: np.ndarray) -> float:
        numerator = 0.0
        denominator = 0.0

        for i in prange(I.shape[0]):
            for j in range(I.shape[1]):
                val = I[i, j][0]
                numerator += val * val
                denominator += val

        return numerator / (denominator * denominator)

    # Used for updating the slm control image, SLM representation on
    # the screen, and the sensor data
    def update_control_image(self, image: np.ndarray) -> None:
        display_image = image.astype(np.uint8)
        cv2.imshow(self.slm.window_name, display_image)
        display_image = cv2.applyColorMap(display_image, cv2.COLORMAP_VIRIDIS)
        cv2.imshow(self.display.window_name, display_image)

        display_image = I.astype(np.uint8)
        display_image = cv2.applyColorMap(display_image, cv2.COLORMAP_VIRIDIS)
        cv2.imshow(self.sensor.window_name, display_image)

def main() -> None:
    with Camera() as cam:
        cam.print_camera_info()
        cam.setup_camera(
            format          = PixelFormat.Mono8,
            exposure_time   = 1500.0,
            gain            = 0.0
            )

        optimizer = Optimizer(cam)
        while True:
            optimizer.iteration()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting.")
                break
            elif key == ord('s'):
                image = optimizer.image.astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite("slm_pattern.png", image)
                print("SLM pattern saved.")
            elif key == ord('c'):
                image = optimizer.cam.get_frame(timeout_ms=2000).as_numpy_ndarray()
                image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite("sensor_data.png", image)
                print("Sensor data saved.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
