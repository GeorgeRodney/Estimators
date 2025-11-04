import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from typing import Optional

# >-----------------------------------------------------
#
#  Function courtesy of ChatGPT. Copy Paste (TM)
#
# >-----------------------------------------------------

class XYStepper:
    """
    Arrow keys:
      → / d : next
      ← / a : prev
      space : play/pause
      r     : reset to 0
      t     : toggle trajectory tails
      + / - : increase/decrease tail length
    """

    def __init__(self,
                 truPos, measPos, estPos, measTime,
                 title: str = "Filter Behavior (XY Stepper)",
                 tail_len: int = 25,
                 show_tails: bool = True):
        # coerce to arrays of shape (N,2)
        self.tru = np.asarray(truPos, dtype=float).reshape(-1, 2)
        self.meas = np.asarray(measPos, dtype=float).reshape(-1, 2)
        self.est = np.asarray(estPos, dtype=float).reshape(-1, 2)
        self.measTime = np.asarray(measTime, dtype=float).reshape(-1, 1)

        # length sanity (use min length to avoid index errors)
        self.N = min(len(self.tru), len(self.meas), len(self.est))

        self.k = 0
        self.play = False
        self.tail_len = int(tail_len)
        self.show_tails = show_tails

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.axis('equal')

        # Pre-plot handles (so we can update fast)
        (self.tru_traj,)  = self.ax.plot([], [], lw=1, alpha=0.6, label="Truth traj")
        (self.meas_traj,) = self.ax.plot([], [], lw=1, alpha=0.6, label="Measured traj")
        (self.est_traj,)  = self.ax.plot([], [], lw=1, alpha=0.6, label="Filtered traj")

        self.tru_pt  = self.ax.scatter([], [], s=35, label="Truth")
        self.meas_pt = self.ax.scatter([], [], s=25, label="Measured")
        self.est_pt  = self.ax.scatter([], [], s=35, label="Filtered")

        self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                 va="top", ha="left", fontsize=9,
                                 bbox=dict(boxstyle="round", fc="w", alpha=0.6))

        self.ax.legend(loc="upper right")

        # Nice bounds so you’re not constantly autoscaling
        all_xy = np.vstack([self.tru, self.meas, self.est])
        xmin, ymin = np.nanmin(all_xy, axis=0)
        xmax, ymax = np.nanmax(all_xy, axis=0)

        # give it a more generous margin (e.g., 20%)
        margin = 2
        xrange = xmax - xmin
        yrange = ymax - ymin

        self.ax.set_xlim(xmin - margin*xrange, xmax + margin*xrange)
        self.ax.set_ylim(ymin - margin*yrange, ymax + margin*yrange)


        self._connect_events()
        self._draw()

        # start a timer for autoplay; it just calls next when play=True
        self.timer = self.fig.canvas.new_timer(interval=120)  # ms per step
        self.timer.add_callback(self._on_timer)
        self.timer.start()

    def _connect_events(self):
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _set_scatter_point(self, h: PathCollection, xy: np.ndarray):
        if np.any(np.isnan(xy)):
            h.set_offsets(np.empty((0, 2)))
        else:
            h.set_offsets(xy.reshape(1, 2))

    def _draw(self):
        k0 = max(0, self.k - (self.tail_len if self.show_tails else 0))
        sl = slice(k0, self.k + 1)

        # Update tails
        self.tru_traj.set_data(self.tru[sl, 0],  self.tru[sl, 1])
        self.meas_traj.set_data(self.meas[sl, 0], self.meas[sl, 1])
        self.est_traj.set_data(self.est[sl, 0],  self.est[sl, 1])

        # Update current points
        self._set_scatter_point(self.tru_pt,  self.tru[self.k])
        self._set_scatter_point(self.meas_pt, self.meas[self.k])
        self._set_scatter_point(self.est_pt,  self.est[self.k])

        def fmt(x):
            return "nan" if np.isnan(x) else f"{x:.3f}"

        self.text.set_text(
            f"Time= ({fmt(self.measTime[self.k,0])})\n"
            f"k = {self.k+1}/{self.N}\n"
            f"Truth   = ({fmt(self.tru[self.k,0])}, {fmt(self.tru[self.k,1])})\n"
            f"Measured= ({fmt(self.meas[self.k,0])}, {fmt(self.meas[self.k,1])})\n"
            f"Filtered= ({fmt(self.est[self.k,0])}, {fmt(self.est[self.k,1])})"
        )

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('right', 'd'):
            self.step(+1)
        elif event.key in ('left', 'a'):
            self.step(-1)
        elif event.key == ' ':
            self.play = not self.play
        elif event.key == 'r':
            self.k = 0
            self._draw()
        elif event.key == 't':
            self.show_tails = not self.show_tails
            self._draw()
        elif event.key == '+':
            self.tail_len = min(self.tail_len + 5, self.N)
            self._draw()
        elif event.key == '-':
            self.tail_len = max(self.tail_len - 5, 0)
            self._draw()

    def _on_timer(self):
        if self.play:
            self.step(+1)

    def step(self, dk: int):
        self.k = int(np.clip(self.k + dk, 0, self.N - 1))
        self._draw()

# ---- usage ----
# truPos, measPos, estPos are lists/arrays of shape (N,2).
# You likely already have them. Example call:
# XYStepper(truPos, measPos, estPos)
# plt.show()
