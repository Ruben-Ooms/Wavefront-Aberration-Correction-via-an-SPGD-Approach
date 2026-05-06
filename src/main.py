"""
SLM Aberration Correction  —  pixel-direct SPSA + AdaDelta
───────────────────────────────────────────────────────────

Threading model
    Matplotlib requires the GUI event loop to run on the main thread —
    this is a hard constraint of every GUI toolkit (Tk, Qt, GTK).
    Calling matplotlib from a background thread will crash or deadlock.

    The solution is to invert the layout:
        Main thread     → owns matplotlib, runs the dashboard event loop
        Optimizer thread → owns the SLM + camera, runs SPSA iterations

    Two queues connect them:

        data_queue   (optimizer → dashboard)
            Each iteration the optimizer puts a DataPacket namedtuple
            containing the metric, g_scalar, elapsed time, phase image,
            and camera frame.  The main thread drains this queue each
            loop and passes the data to dash.update().

            maxsize=2 so the optimizer never accumulates a backlog of
            stale frames if the dashboard is slow — it just drops the
            oldest and keeps going.

        cmd_queue    (dashboard → optimizer)
            Button/key presses produce string commands ("reset",
            "save_slm", "save_camera", "quit") that the optimizer
            thread checks at the start of each iteration.

Usage
    python main.py

Controls (matplotlib dashboard)
    q  – quit and save state
    s  – save SLM pattern PNG (viridis)
    c  – save camera frame PNG (viridis)
    r  – reset phase to flat
"""

import queue
import threading
import cv2
import time
import numpy as np
import screeninfo
import matplotlib.pyplot as plt
from collections import namedtuple
from numba import njit
from vmbpy import PixelFormat
from sys import exit

from adadelta  import AdaDelta
from camera    import Camera
from cv_window import CVWindow
from dashboard import Dashboard

# ── Terminal Table ───────────────────────────────────────────

# Set sign properly
def _trend(current: float, previous: float | None) -> str:
    if previous is None:
        return "  —"
    delta = current - previous
    if abs(delta) < 1e-6:
        return " ="
    if delta > 0:
        return f" +{delta:.4f}"
    return f" {delta:.4f}"

# Terminal table header
def print_header():
    w = 54
    print()
    print(f"{'SLM Aberration Correction':^{w}}")
    print("─" * w)
    print(f"{'Iter':^6}  {'Metric':^8}  {'Trend':^7} {'δJ':^4}  {'Time':^6}")
    print("─" * w)

# Terminal table row
def print_iter(k: int, metric: float, prev_metric: float | None,
               g_scalar: float, elapsed: float):
    trend = _trend(metric, prev_metric)
    print(
        f"  {k:>4}  "
        f"{metric:.4f}  "
        f"{trend:>7}  "
        f"{g_scalar:>+.4f}  "
        f"{elapsed:4.2f}s"
    )


# ── Thread data packet ───────────────────────────────────────

DataPacket = namedtuple("DataPacket", ["metric", "g_scalar", "elapsed", "phase_gray", "frame", "roi"])


# ── ROI Detection ────────────────────────────────────────────

ROI_UPDATE_INTERVAL = 5   # re-detect PSF location every N iterations

# Finds a square ROI centered at the brightest region of the fram
# If silent = True then it does not print
def detect_roi(frame: np.ndarray,
               roi_size: int = 64,
               silent: bool = True) -> tuple[int, int, int, int]:
    img     = frame.squeeze().astype(np.float32)
    H, W    = img.shape
    half    = roi_size // 2
    kernel  = np.ones((roi_size, roi_size), dtype=np.float32) / roi_size ** 2
    blurred = cv2.filter2D(img, -1, kernel)
    cy, cx  = np.unravel_index(np.argmax(blurred), blurred.shape)
    x0 = int(np.clip(cx - half, 0, W - roi_size))
    y0 = int(np.clip(cy - half, 0, H - roi_size))
    if not silent:
        print(f"ROI center=({int(cx)}, {int(cy)}), box between ({x0}, {y0}) to ({x0+roi_size}, {y0+roi_size})")
    return x0, y0, x0 + roi_size, y0 + roi_size


# ── Metric ─────────────────────────────────────────────────

# Computes the normzalized peak intensity inside of the ROI
@njit(fastmath=True, cache=True)
def compute_metric(roi: np.ndarray) -> float:
    peak = 0.0
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            v = float(roi[i, j])
            if v > peak:
                peak = v
    return peak / 255.0

def extract_roi(frame: np.ndarray,
                roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return frame.squeeze()[y0:y1, x0:x1]


# ── Optimizer ────────────────────────────────────────────────


class Optimizer:

    def __init__(self, cam: Camera,
                 work_scale:   float = 0.25,
                 spsa_delta:   float = 16.0,
                 settle_ms:    int   = 60,
                 adadelta_rho: float = 0.9,
                 roi_size:     int   = 64):

        monitors = screeninfo.get_monitors()
        if len(monitors) < 2:
            print(f"ERROR: SLM not detected (need ≥2 monitors).")
            exit(1)

        self.win_slm   = CVWindow(monitors[1], "SLM", fullscreen=True)
        self.W, self.H = monitors[1].width, monitors[1].height
        print(f"SLM  {self.W}×{self.H}")

        self.wW = max(1, int(self.W * work_scale))
        self.wH = max(1, int(self.H * work_scale))
        print(f"Working resolution {self.wW}×{self.wH} ({self.wW * self.wH:,} params)")

        self.phase      = np.full((self.wH, self.wW), 128.0, dtype=np.float32)
        self.spsa_delta = spsa_delta
        self.settle_ms  = settle_ms
        self.roi_size   = roi_size

        self._dz   = np.zeros((self.wH, self.wW), dtype=np.float32)
        self._grad = np.zeros((self.wH, self.wW), dtype=np.float32)

        self.opt = AdaDelta((self.wH, self.wW), rho=adadelta_rho)
        self.cam = cam
        self.k   = 0
        self._prev_metric: float | None = None
        self.history: list[float] = []

        _ = compute_metric(np.ones((4, 4), dtype=np.uint8))

        self._push(self.phase)
        print(f"Detecting PSF location...")
        self.roi = detect_roi(self.cam.grab(), roi_size=roi_size, silent=False)

    # ── Internal ─────────────────────────────────────────────

    def _to_slm(self, phase_work: np.ndarray) -> np.ndarray:
        full = cv2.resize(phase_work, (self.W, self.H),
                          interpolation=cv2.INTER_CUBIC)
        return np.clip(full, 0, 255).astype(np.uint8)

    def _push(self, phase_work: np.ndarray) -> np.ndarray:
        gray = self._to_slm(phase_work)
        self.win_slm.show(gray)
        time.sleep(self.settle_ms / 1000.0)
        return gray

    def _push_and_grab(self, phase_work: np.ndarray) -> tuple[float, np.ndarray]:
        self._push(phase_work)
        frame = self.cam.grab()
        return float(compute_metric(extract_roi(frame, self.roi))), frame

    def _fill_perturbation(self):
        signs = (2 * np.random.randint(0, 2, (self.wH, self.wW)) - 1
                 ).astype(np.float32)
        np.multiply(signs, self.spsa_delta, out=self._dz)

    def detect_roi(self):
        self.roi = detect_roi(self.cam.grab(), roi_size=self.roi_size, Silent=False)

    # ── Public ───────────────────────────────────────────────

    # One iteration runs a positive and negative perturbation
    def iteration(self) -> DataPacket:
        t0 = time.perf_counter()

        self._fill_perturbation()
        J_plus,  frame_plus  = self._push_and_grab(
            np.clip(self.phase + self._dz, 0, 255))
        J_minus, frame_minus = self._push_and_grab(
            np.clip(self.phase - self._dz, 0, 255))

        g_scalar = J_plus - J_minus
        np.multiply(-g_scalar, self._dz, out=self._grad)
        np.clip(self.opt.step(self.phase, self._grad), 0.0, 255.0,
                out=self.phase)

        self._push(self.phase)

        if J_plus >= J_minus:
            m, last_frame = J_plus, frame_plus
        else:
            m, last_frame = J_minus, frame_minus

        # re-detect ROI from the latest frame every N iterations
        if self.k % ROI_UPDATE_INTERVAL == 0:
            self.roi = detect_roi(last_frame, self.roi_size)

        self.history.append(m)
        self.k += 1
        elapsed = time.perf_counter() - t0

        print_iter(self.k, m, self._prev_metric, g_scalar, elapsed)
        self._prev_metric = m

        return DataPacket(
            metric     = m,
            g_scalar   = g_scalar,
            elapsed    = elapsed,
            phase_gray = self._to_slm(self.phase),
            frame      = last_frame,
            roi        = self.roi,
        )

    def reset(self):
        self.phase[:] = 128.0
        self._grad[:] = 0.0
        self._prev_metric = None
        self.history.clear()
        self.k = 0
        self.opt.reset()
        self._push(self.phase)
        print(f"\nReset to flat wavefront.")
        print_header()

    def save_slm_png(self, path: str = "slm_pattern.png"):
        cv2.imwrite(path, cv2.applyColorMap(
            self._to_slm(self.phase), cv2.COLORMAP_VIRIDIS))
        print(f"{path} saved")

    def save_camera_png(self, path: str = "camera_frame.png",
                        frame: np.ndarray | None = None):
        if frame is None:
            frame = self.cam.grab()
        cv2.imwrite(path, cv2.applyColorMap(
            frame.squeeze().astype(np.uint8), cv2.COLORMAP_VIRIDIS))
        print(f"{path} saved")


# ── Optimizer Threading ──────────────────────────────────────

# Runs the optimizer on another thread
def optimizer_loop(opt: Optimizer,
                   data_queue: queue.Queue,
                   cmd_queue:  queue.Queue,
                   stop_event: threading.Event):
    print_header()

    while not stop_event.is_set():
        # ── Handle Commands ──────────────────────────────────
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "reset":
                opt.reset()
            elif cmd == "save_slm":
                opt.save_slm_png()
            elif cmd == "save_camera":
                opt.save_camera_png()
            elif cmd == "quit":
                stop_event.set()
                return

        # ── Optimisation Iteration ───────────────────────────
        try:
            packet = opt.iteration()
        except Exception as e:
            print(f"\nERROR in optimizer thread: {e}")
            stop_event.set()
            return

        # put packet; drop oldest if full
        try:
            data_queue.put_nowait(packet)
        except queue.Full:
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass
            data_queue.put_nowait(packet)


# ── Main ─────────────────────────────────────────────────────

def main():
    try:
        with Camera() as cam:
            cam.print_info()
            cam.setup(fmt=PixelFormat.Mono8, exposure_us=250.0, gain=0.0)

            opt = Optimizer(
                cam,
                work_scale   = 0.25,
                spsa_delta   = 16.0,
                settle_ms    = 5,
                adadelta_rho = 0.9,
                roi_size     = 256,
            )

            data_queue = queue.Queue(maxsize=2)
            cmd_queue  = queue.Queue()
            stop_event = threading.Event()

            dash = Dashboard(opt.roi)

            def _send(cmd):
                cmd_queue.put(cmd)

            dash.btn_save.on_clicked( lambda _: _send("save_slm"))
            dash.btn_cam.on_clicked(  lambda _: _send("save_camera"))
            dash.btn_reset.on_clicked(lambda _: _send("reset"))
            dash.btn_quit.on_clicked( lambda _: _send("quit"))

            def _key(event):
                if   event.key == "q": _send("quit")
                elif event.key == "s": _send("save_slm")
                elif event.key == "c": _send("save_camera")
                elif event.key == "r": _send("reset")
            dash.fig.canvas.mpl_connect("key_press_event", _key)

            print(f"\nDashboard: q=quit  s=save SLM c=save camera  r=reset")

            opt_thread = threading.Thread(
                target = optimizer_loop,
                args   = (opt, data_queue, cmd_queue, stop_event),
                daemon = True,
                name   = "optimizer",
            )
            opt_thread.start()

            # ── Matplotlib Thread in Main Loop ───────────────
            while not stop_event.is_set():
                packet = None
                while True:
                    try:
                        packet = data_queue.get_nowait()
                    except queue.Empty:
                        break

                if packet is not None:
                    dash.roi = packet.roi   # keep dashboard ROI in sync
                    dash.update(packet.metric, packet.phase_gray, packet.frame)

                dash.fig.canvas.flush_events()
                plt.pause(0.01)

            opt_thread.join(timeout=1.0)

            print(f"\nDashboard paused. Close the dashboard to EXIT the program.")

    except KeyboardInterrupt:
        print(f"\nInterrupted.")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
