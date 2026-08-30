import shutil
import subprocess
import threading
import time

# ── GNOME hot-corner guard ────────────────────────────────────────────────────
HOT_CORNER_MARGIN = 20  # pixels — cursor is clamped this far from corners

# ── evdev backend ─────────────────────────────────────────────────────────────
try:
    from evdev import UInput, ecodes

    _EVDEV_IMPORTABLE = True
except ImportError:
    _EVDEV_IMPORTABLE = False


def _read_cursor_pos_once() -> tuple[int | None, int | None]:
    """Read current cursor position via xdotool (called once at startup)."""
    try:
        out = subprocess.check_output(
            ["xdotool", "getmouselocation"], timeout=1.0, stderr=subprocess.DEVNULL
        ).decode()
        x = int(out.split("x:")[1].split()[0])
        y = int(out.split("y:")[1].split()[0])
        return x, y
    except Exception:
        return None, None


class _EvdevMouse:
    """
    Virtual relative mouse via uinput.
    Tracks own absolute position; converts move_absolute() to REL deltas.
    This is exactly how ydotoold drives the cursor internally.
    """

    def __init__(self, screen_w: int, screen_h: int):
        rel_caps = [ecodes.REL_X, ecodes.REL_Y, ecodes.REL_WHEEL]
        if hasattr(ecodes, "REL_WHEEL_HI_RES"):
            rel_caps.append(ecodes.REL_WHEEL_HI_RES)
        caps = {
            ecodes.EV_REL: rel_caps,
            ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE],
        }
        self.ui = UInput(caps, name="telekinesis-virtual-mouse")
        cx, cy = _read_cursor_pos_once()
        self._x = cx if cx is not None else screen_w // 2
        self._y = cy if cy is not None else screen_h // 2
        self._wheel_accum: float = 0.0  # accumulates sub-notch scroll for REL_WHEEL

    # ── motion ────────────────────────────────────────────────────────────────

    def move(self, x: int, y: int):
        dx, dy = x - self._x, y - self._y
        if dx == 0 and dy == 0:
            return
        if dx != 0:
            self.ui.write(ecodes.EV_REL, ecodes.REL_X, dx)
        if dy != 0:
            self.ui.write(ecodes.EV_REL, ecodes.REL_Y, dy)
        self.ui.syn()
        self._x, self._y = x, y

    # ── buttons ───────────────────────────────────────────────────────────────

    def click_left(self):
        self.ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
        self.ui.syn()
        time.sleep(0.04)
        self.ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
        self.ui.syn()

    def mouse_down(self):
        self.ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
        self.ui.syn()

    def mouse_up(self):
        self.ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
        self.ui.syn()

    # ── scroll ────────────────────────────────────────────────────────────────

    def scroll_wheel(self, notches: float):
        """
        notches > 0 → scroll up, < 0 → scroll down.  notches is per-frame
        (typically 0.05 – 0.3 at 30fps).

        REL_WHEEL_HI_RES: 120 units = 1 notch.  Modern GTK/Qt apps use this
        for smooth continuous scrolling (pixel-level inertia).

        REL_WHEEL: integer notches.  Apps that only listen to this event get
        a tick whenever the fractional accumulator crosses a whole notch.
        """
        if notches == 0.0:
            return

        # HI_RES path — most modern apps
        hi_res = int(notches * 120)
        if hi_res != 0 and hasattr(ecodes, "REL_WHEEL_HI_RES"):
            self.ui.write(ecodes.EV_REL, ecodes.REL_WHEEL_HI_RES, hi_res)

        # Accumulate fractional notches for legacy apps that only read REL_WHEEL
        self._wheel_accum += notches
        coarse = int(self._wheel_accum)  # whole notches crossed so far
        if coarse != 0:
            self.ui.write(ecodes.EV_REL, ecodes.REL_WHEEL, coarse)
            self._wheel_accum -= coarse

        self.ui.syn()

    def close(self):
        self.ui.close()


# ── ydotool backend ───────────────────────────────────────────────────────────
_YDOTOOL_BIN = shutil.which("ydotool")

if _YDOTOOL_BIN is None:
    print("WARNING: `ydotool` not found. Install it and start `ydotoold` as fallback.")


def _ydotool_run(args: list[str]):
    if _YDOTOOL_BIN is None:
        return
    try:
        subprocess.run([_YDOTOOL_BIN] + args, check=False, timeout=0.5)
    except (subprocess.TimeoutExpired, OSError):
        pass


class _YdotoolCursorWorker(threading.Thread):
    """Cursor movement off the main thread. Drops stale positions."""

    def __init__(self):
        super().__init__(daemon=True, name="ydotool-cursor")
        self._lock = threading.Lock()
        self._pending: tuple[int, int] | None = None
        self._event = threading.Event()

    def move(self, x: int, y: int):
        with self._lock:
            self._pending = (x, y)
        self._event.set()

    def run(self):
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                pos, self._pending = self._pending, None
            if pos and _YDOTOOL_BIN:
                x, y = pos
                try:
                    subprocess.run(
                        [_YDOTOOL_BIN, "mousemove", str(x), str(y)],
                        check=False,
                        timeout=0.5,
                    )
                except (subprocess.TimeoutExpired, OSError):
                    pass


# ── Window mover ──────────────────────────────────────────────────────────────


class _WindowMoverWorker(threading.Thread):
    """
    Moves X11 / XWayland windows via xdotool windowmove in a background
    thread.  Drops stale positions so the inference loop is never blocked.
    """

    def __init__(self):
        super().__init__(daemon=True, name="window-mover")
        self._lock = threading.Lock()
        self._pending: tuple[int, int, int] | None = None  # (wid, x, y)
        self._event = threading.Event()

    def move(self, window_id: int, x: int, y: int):
        with self._lock:
            self._pending = (window_id, x, y)
        self._event.set()

    def run(self):
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            return
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job:
                wid, x, y = job
                try:
                    subprocess.run(
                        [xdotool, "windowmove", str(wid), str(x), str(y)],
                        check=False,
                        timeout=0.15,
                    )
                except (subprocess.TimeoutExpired, OSError):
                    pass


# ── Module-level state ────────────────────────────────────────────────────────
_evdev_mouse: "_EvdevMouse | None" = None
_ydotool_worker: "_YdotoolCursorWorker | None" = None
_window_mover: "_WindowMoverWorker | None" = None
_screen_w = 1920
_screen_h = 1080


def init_cursor_backend(screen_w: int, screen_h: int):
    """
    Call once at startup.  Selects evdev or ydotool, starts window mover.
    """
    global _evdev_mouse, _ydotool_worker, _window_mover, _screen_w, _screen_h
    _screen_w, _screen_h = screen_w, screen_h

    if _EVDEV_IMPORTABLE:
        try:
            _evdev_mouse = _EvdevMouse(screen_w, screen_h)
            print("✓ Cursor backend: evdev uinput  (zero-latency)")
        except PermissionError:
            print(
                "  evdev: /dev/uinput denied — run: sudo usermod -aG input $USER && newgrp input"
            )
        except Exception as e:
            print(f"  evdev: init failed ({e})")

    if _evdev_mouse is None:
        if _YDOTOOL_BIN:
            _ydotool_worker = _YdotoolCursorWorker()
            _ydotool_worker.start()
            print("✓ Cursor backend: threaded ydotool")
        else:
            print("✗ No cursor backend. Install evdev (preferred) or ydotool.")

    # Window mover always available (uses xdotool independently)
    if shutil.which("xdotool"):
        _window_mover = _WindowMoverWorker()
        _window_mover.start()
        print("✓ Window drag: xdotool  (X11 / XWayland apps)")
    else:
        print("  Window drag unavailable — install xdotool.")


# ── Cursor ────────────────────────────────────────────────────────────────────


def _clamp(x: int, y: int) -> tuple[int, int]:
    m = HOT_CORNER_MARGIN
    return max(m, min(_screen_w - m, x)), max(m, min(_screen_h - m, y))


def move_cursor_absolute(x: int, y: int):
    x, y = _clamp(x, y)
    if _evdev_mouse:
        _evdev_mouse.move(x, y)
    elif _ydotool_worker:
        _ydotool_worker.move(x, y)


def click_left():
    if _evdev_mouse:
        _evdev_mouse.click_left()
    else:
        _ydotool_run(["click", "0xC0"])


def mouse_down():
    if _evdev_mouse:
        _evdev_mouse.mouse_down()
    else:
        _ydotool_run(["click", "0x40"])


def mouse_up():
    if _evdev_mouse:
        _evdev_mouse.mouse_up()
    else:
        _ydotool_run(["click", "0x80"])


def scroll_wheel(notches: float):
    if _evdev_mouse:
        _evdev_mouse.scroll_wheel(notches)
    elif notches > 0:
        _ydotool_run(["key", "Up"])
    elif notches < 0:
        _ydotool_run(["key", "Down"])


# ── Keyboard ──────────────────────────────────────────────────────────────────


def key_press(key_name: str):
    _ydotool_run(["key", key_name])


def zoom_in():
    _ydotool_run(["key", "ctrl+plus"])


def zoom_out():
    _ydotool_run(["key", "ctrl+minus"])


def swipe_next():
    _ydotool_run(["key", "Right"])


def swipe_prev():
    _ydotool_run(["key", "Left"])


# ── Window drag ───────────────────────────────────────────────────────────────


def get_window_under_cursor() -> tuple[int | None, int, int]:
    """
    Returns (window_id, window_x, window_y) for the X11/XWayland window
    currently under the cursor.  Called ONCE at fist-gesture entry.
    Returns (None, 0, 0) if xdotool is unavailable or the window is a
    pure-Wayland surface (which xdotool cannot query).
    """
    xdotool = shutil.which("xdotool")
    if xdotool is None:
        return None, 0, 0
    try:
        # Get window ID at cursor
        loc = subprocess.check_output(
            [xdotool, "getmouselocation", "--shell"], timeout=1.0
        ).decode()
        wid = None
        for line in loc.splitlines():
            if line.startswith("WINDOW="):
                wid_str = line.split("=")[1].strip()
                if wid_str and wid_str != "0":
                    wid = int(wid_str)
                break
        if wid is None:
            return None, 0, 0

        # Get window top-left position
        geo = subprocess.check_output(
            [xdotool, "getwindowgeometry", "--shell", str(wid)], timeout=1.0
        ).decode()
        wx = wy = 0
        for line in geo.splitlines():
            if line.startswith("X="):
                wx = int(line.split("=")[1])
            elif line.startswith("Y="):
                wy = int(line.split("=")[1])
        return wid, wx, wy

    except Exception:
        return None, 0, 0


def move_window(window_id: int, x: int, y: int):
    """Enqueue a window move (non-blocking, drops stale positions)."""
    if _window_mover and window_id:
        _window_mover.move(window_id, max(0, x), max(0, y))


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess as _sp

    _sw, _sh = 1920, 1080
    try:
        out = _sp.check_output(["xrandr"]).decode()
        for line in out.splitlines():
            if " connected" in line:
                for tok in line.split():
                    if "x" in tok and tok[0].isdigit():
                        w, h = tok.split("+")[0].split("x")
                        _sw, _sh = int(w), int(h)
                        break
    except Exception:
        pass

    print(f"Screen: {_sw}x{_sh}")
    init_cursor_backend(_sw, _sh)
    time.sleep(0.2)
    move_cursor_absolute(_sw // 2, _sh // 2)
    time.sleep(0.3)
    scroll_wheel(2.0)
    time.sleep(0.1)
    scroll_wheel(-2.0)
    time.sleep(0.1)
    mouse_down()
    time.sleep(0.05)
    mouse_up()
    print("Done.")
