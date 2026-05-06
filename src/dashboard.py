import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Defines a dashboard
# Top left is the metric history
# Top right is the SLM phase histogram
# Bottom left is the SLM phase map
# Bottom right is the sensor data
class Dashboard:

    SLOW_INTERVAL = 5 # Histogram + phase map every N iterations

    def __init__(self, roi: tuple[int, int, int, int]):
        self.roi             = roi
        self._iter           = 0
        self._hist: list[float] = []
        self._cam_h          = 240
        self._cam_w          = 320

        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.40, wspace=0.32,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.13)

        ax_kw  = dict(facecolor="#16213e")
        lbl_kw = dict(color="#e0e0e0", fontsize=9)

        # ── Metric History ───────────────────────────────────
        self.ax_m = self.fig.add_subplot(gs[0, 0], **ax_kw)
        self.ax_m.set_title("Metric",
                             color="#e0e0e0", fontsize=10)
        self.ax_m.set_xlabel("Iteration", **lbl_kw)
        self.ax_m.set_ylabel("Normalized Peak Intensity", **lbl_kw)
        self.ax_m.set_ylim(0, 1)
        self.ax_m.tick_params(colors="#888")
        for sp in self.ax_m.spines.values():
            sp.set_color("#333")
        self.ln_m,  = self.ax_m.plot([], [], color="#00d4ff", lw=1.0,
                                     label="current")
        self.ln_sm20, = self.ax_m.plot([], [], color="#ff9f43", lw=1.8,
                                     ls="--", alpha=0.8, label="smoothed 20")
        self.ln_sm50, = self.ax_m.plot([], [], color="#3ffb2b", lw=1.8,
                                     ls="--", alpha=0.8, label="smoothed 50")
        self.ax_m.legend(fontsize=8, facecolor="#16213e",
                         labelcolor="#e0e0e0", edgecolor="#333")

        # ── Phase Histogram ──────────────────────────────────
        self.ax_h = self.fig.add_subplot(gs[0, 1], **ax_kw)
        self.ax_h.set_title(f"Phase Distribution [every {self.SLOW_INTERVAL} iters]",
                            color="#e0e0e0", fontsize=10)
        self.ax_h.set_xlabel("Grey level (0–255)", **lbl_kw)
        self.ax_h.set_ylabel("Pixel count", **lbl_kw)
        self.ax_h.tick_params(colors="#888")
        for sp in self.ax_h.spines.values():
            sp.set_color("#333")
        self.hist_bars = self.ax_h.bar(np.arange(256), np.zeros(256),
                                       width=1.0, color="#7b68ee",
                                       edgecolor="none", align="edge")

        # ── SLM Phase Map ────────────────────────────────────
        self.ax_slm = self.fig.add_subplot(gs[1, 0], **ax_kw)
        self.ax_slm.set_title(f"SLM phase map [every {self.SLOW_INTERVAL} iters]",
                              color="#e0e0e0", fontsize=10)
        self.ax_slm.axis("off")
        self.im_slm = self.ax_slm.imshow(
            np.full((100, 100), 128, dtype=np.uint8),
            # cmap="viridis", vmin=0, vmax=255, aspect="auto")
            cmap="viridis", aspect="auto")
        self.fig.colorbar(self.im_slm, ax=self.ax_slm,
                          fraction=0.046, pad=0.04).ax.tick_params(
                              colors="#888", labelsize=7)

        # ── Camera Feed ──────────────────────────────────────
        self.ax_cam = self.fig.add_subplot(gs[1, 1], **ax_kw)
        self.ax_cam.set_title("Sensor (cyan box = ROI)",
                               color="#e0e0e0", fontsize=10)
        self.ax_cam.axis("off")

        dummy = np.zeros((self._cam_h, self._cam_w), dtype=np.uint8)
        self.im_cam = self.ax_cam.imshow(
            dummy, cmap="viridis", vmin=0, vmax=255, aspect="auto",
            extent=[0, self._cam_w, self._cam_h, 0])
        self.fig.colorbar(self.im_cam, ax=self.ax_cam,
                          fraction=0.046, pad=0.04).ax.tick_params(
                              colors="#888", labelsize=7)

        # ── Roi Ontop of Camera Feed ─────────────────────────
        self._roi_rect = patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=1.5, edgecolor="cyan", facecolor="none",
            linestyle="--", zorder=10)
        self.ax_cam.add_patch(self._roi_rect)

        self.fig.suptitle(
            "SLM Wavefront Correction using AdaDelta",
            color="#e0e0e0", fontsize=11)

        # ── Buttons ──────────────────────────────────────────
        btn_kw = dict(color="#2a2a4a", hovercolor="#3a3a6a")
        by, bh = 0.03, 0.06
        axes = [self.fig.add_axes([x, by, 0.17, bh])
                for x in (0.10, 0.31, 0.52, 0.73)]
        labels = ["Save SLM  [s]", "Save Camera  [c]",
                  "Reset  [r]",    "Quit  [q]"]
        self.btn_save, self.btn_cam, self.btn_reset, self.btn_quit = [
            Button(ax, lbl, **btn_kw) for ax, lbl in zip(axes, labels)]

        for btn in (self.btn_save, self.btn_cam, self.btn_reset, self.btn_quit):
            btn.label.set_color("#e0e0e0")
            btn.label.set_fontsize(9)

        # Button callbacks and key bindings are wired externally in main.py
        # via cmd_queue so they are thread-safe. Dashboard itself holds
        # no command state — it is purely a display component.
        plt.pause(0.05)

    # ── ROI Rectangle ────────────────────────────────────────

    def _update_roi_rect(self, frame_h: int, frame_w: int,
                         thumb_h: int, thumb_w: int):
        x0, y0, x1, y1 = self.roi
        sx = thumb_w / frame_w
        sy = thumb_h / frame_h
        self._roi_rect.set_xy((x0 * sx, y0 * sy))
        self._roi_rect.set_width( (x1 - x0) * sx)
        self._roi_rect.set_height((y1 - y0) * sy)

    # ── Update ───────────────────────────────────────────────

    def update(self, metric: float,
               phase_full: np.ndarray,
               camera_frame: np.ndarray):
        self._iter += 1
        self._hist.append(metric)

        # always flush — keeps buttons and keys alive with minimal cost
        self.fig.canvas.flush_events()

        do_slow   = (self._iter % self.SLOW_INTERVAL   == 0)

        # ── Camera ───────────────────────────────────────────
        if camera_frame is not None:
            f      = camera_frame.squeeze()
            fH, fW = f.shape
            tW     = min(fW, self._cam_w)
            tH     = min(fH, self._cam_h)
            fc     = cv2.resize(f, (tW, tH))

            if (tH, tW) != (self._cam_h, self._cam_w):
                self._cam_h, self._cam_w = tH, tW
                self.im_cam.set_extent([0, tW, tH, 0])

            self.im_cam.set_data(fc)
            self.im_cam.set_clim(0, max(int(fc.max()), 1))
            self._update_roi_rect(fH, fW, tH, tW)

        # ── Metric Lines ─────────────────────────────────────

        iters = np.arange(len(self._hist))
        self.ln_m.set_data(iters, self._hist)
        window1 = 20
        window2 = 50
        if len(self._hist) >= window1:
            smoothed = np.convolve(
                self._hist, np.ones(window1) / window1, mode="valid")
            self.ln_sm20.set_data(iters[window1 - 1:], smoothed)
        if len(self._hist) >= window2:
            smoothed = np.convolve(
                self._hist, np.ones(window2) / window2, mode="valid")
            self.ln_sm50.set_data(iters[window2 - 1:], smoothed)
        self.ax_m.relim()
        self.ax_m.autoscale_view()

        # ── Slow Panels ──────────────────────────────────────
        if do_slow:
            flat   = phase_full[::4, ::4].ravel().astype(np.uint8)
            counts = np.bincount(flat, minlength=256).astype(float)
            for bar, h in zip(self.hist_bars, counts):
                bar.set_height(h)
            self.ax_h.set_ylim(0, counts.max() * 1.1 + 1)

            thumb = cv2.resize(phase_full,
                               (min(phase_full.shape[1], 320),
                                min(phase_full.shape[0], 240)))
            self.im_slm.set_data(thumb)

        plt.pause(0.001)
