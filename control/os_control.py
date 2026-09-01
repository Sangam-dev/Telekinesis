import shutil
import subprocess
import threading
import time

# ── GNOME hot-corner guard ────────────────────────────────────────────────────
HOT_CORNER_MARGIN = 20  # pixels — cursor is clamped this far from corners

# ── evdev backend ─────────────────────────────────────────────────────────────
try:
    from evdev import UInput, AbsInfo, ecodes

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
    Virtual ABSOLUTE-positioning mouse via uinput.

    Earlier this sent REL_X/REL_Y deltas and just assumed the OS applied
    them 1:1 to the real cursor — but libinput applies pointer acceleration
    to relative motion by default, so a requested delta and the actual
    on-screen movement don't always match. That per-frame error compounds
    forever, so the "target" position (what we think we told the cursor)
    and the real cursor position drift apart over time — which is exactly
    why the orb ended up "far from the actual pointer."

    Using EV_ABS (ABS_X/ABS_Y) instead of EV_REL sidesteps the problem
    entirely: an absolute-positioning device (marked INPUT_PROP_DIRECT, the
    same property a touchscreen uses) tells the OS exactly which pixel the
    pointer should be at, with no acceleration curve applied. The commanded
    position and the real position become the same thing by construction,
    every single frame — no drift, no resync needed, no extra polling
    latency.
    """

    def __init__(self, screen_w: int, screen_h: int):
        self._sw, self._sh = screen_w, screen_h
        rel_caps = [ecodes.REL_WHEEL]
        if hasattr(ecodes, "REL_WHEEL_HI_RES"):
            rel_caps.append(ecodes.REL_WHEEL_HI_RES)

        abs_caps = [
            (ecodes.ABS_X, AbsInfo(value=0, min=0, max=screen_w - 1, fuzz=0, flat=0, resolution=0)),
            (ecodes.ABS_Y, AbsInfo(value=0, min=0, max=screen_h - 1, fuzz=0, flat=0, resolution=0)),
        ]
        caps = {
            ecodes.EV_ABS: abs_caps,
            ecodes.EV_REL: rel_caps,
            ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE],
        }

        # INPUT_PROP_DIRECT tells libinput this is a direct-positioning
        # device (like a touchscreen), not a relative tablet needing its
        # own calibration matrix — required for ABS events to map 1:1 to
        # screen pixels. Falls back gracefully on older python-evdev that
        # doesn't support the input_props kwarg.
        try:
            self.ui = UInput(
                caps,
                name="telekinesis-virtual-mouse",
                input_props=[ecodes.INPUT_PROP_DIRECT],
            )
        except TypeError:
            self.ui = UInput(caps, name="telekinesis-virtual-mouse")

        cx, cy = _read_cursor_pos_once()
        self._x = cx if cx is not None else screen_w // 2
        self._y = cy if cy is not None else screen_h // 2
        _tracked_cursor[0] = self._x
        _tracked_cursor[1] = self._y
        self._wheel_accum: float = 0.0  # accumulates sub-notch scroll for REL_WHEEL

    # ── motion ────────────────────────────────────────────────────────────────

    def move(self, x: int, y: int):
        x = min(max(x, 0), self._sw - 1)
        y = min(max(y, 0), self._sh - 1)
        if x == self._x and y == self._y:
            return
        self.ui.write(ecodes.EV_ABS, ecodes.ABS_X, x)
        self.ui.write(ecodes.EV_ABS, ecodes.ABS_Y, y)
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
                    # `ydotool mousemove` is RELATIVE by default — it was
                    # being called with our absolute pixel targets as if
                    # they were deltas, which sends the cursor to the wrong
                    # place almost every frame. --absolute makes it treat
                    # x, y as the actual absolute screen position we mean.
                    subprocess.run(
                        [_YDOTOOL_BIN, "mousemove", "--absolute", str(x), str(y)],
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


# ── Real cursor-position reader ────────────────────────────────────────────────


class _CursorPosReader(threading.Thread):
    """
    Continuously polls the REAL pointer position (via xdotool) in a background
    thread and keeps `_tracked_cursor` in sync with what is actually on screen.

    This lets the neon orb be positioned EXACTLY on the real cursor (replacing
    it) rather than relying on our own internal deltas, which can drift a few
    pixels/centimetres from the OS pointer.
    """

    def __init__(self, poll_interval: float = 0.03):
        super().__init__(daemon=True, name="cursor-pos-reader")
        self._poll = poll_interval
        self._xdotool = shutil.which("xdotool")

    def run(self):
        global _tracked_cursor, _screen_w, _screen_h
        if self._xdotool is None:
            # No xdotool — just leave tracked cursor at whatever evdev believes.
            return
        while True:
            try:
                out = subprocess.check_output(
                    [self._xdotool, "getmouselocation"],
                    timeout=0.2,
                    stderr=subprocess.DEVNULL,
                ).decode()
                x = int(out.split("x:")[1].split()[0])
                y = int(out.split("y:")[1].split()[0])
                x = min(max(x, 0), _screen_w - 1)
                y = min(max(y, 0), _screen_h - 1)
                _tracked_cursor[0] = x
                _tracked_cursor[1] = y

                # Keep the evdev backend's internal (self._x, self._y)
                # bookkeeping (used only to skip redundant no-op writes)
                # in sync with the real, observed position. With absolute
                # positioning this is just a safety net (e.g. if something
                # external moves the pointer) rather than a drift fix.
                if _evdev_mouse is not None:
                    _evdev_mouse._x = x
                    _evdev_mouse._y = y
            except Exception:
                pass
            time.sleep(self._poll)


# ── Module-level state ────────────────────────────────────────────────────────
_evdev_mouse: "_EvdevMouse | None" = None
_ydotool_worker: "_YdotoolCursorWorker | None" = None
_window_mover: "_WindowMoverWorker | None" = None
_cursor_pos_reader: "_CursorPosReader | None" = None
_screen_w = 1920
_screen_h = 1080
_tracked_cursor = [1920 // 2, 1080 // 2]  # [x, y] updated on every cursor move


def get_cursor_position() -> tuple[int, int]:
    """Latest absolute cursor position (in screen pixels). Used by the overlay."""
    return int(_tracked_cursor[0]), int(_tracked_cursor[1])


def init_cursor_backend(screen_w: int, screen_h: int):
    """
    Call once at startup.  Selects evdev or ydotool, starts window mover.
    """
    global _evdev_mouse, _ydotool_worker, _window_mover, _screen_w, _screen_h
    _screen_w, _screen_h = screen_w, screen_h

    # Real-pointer reader that keeps the orb locked onto the actual cursor.
    if shutil.which("xdotool"):
        _cursor_pos_reader = _CursorPosReader()
        _cursor_pos_reader.start()

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
    _tracked_cursor[0] = x
    _tracked_cursor[1] = y
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