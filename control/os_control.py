import subprocess
import shutil

_YDOTOOL_BIN = shutil.which("ydotool")

if _YDOTOOL_BIN is None:
    print("WARNING: `ydotool` binary not found on PATH. Install it and ensure `ydotoold` is running.")


def _run(args):
    if _YDOTOOL_BIN is None:
        return
    try:
        subprocess.run([_YDOTOOL_BIN] + args, check=False, timeout=1.0)
    except subprocess.TimeoutExpired:
        print("ydotool call timed out — is ydotoold running?")


def move_cursor_absolute(x: int, y: int):
    """Move cursor to absolute screen pixel coordinates."""
    _run(["mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))])


def click_left():
    _run(["click", "0xC0"])


def mouse_down_left():
    _run(["click", "--", "0x40"])  # press-only (ydotool button-down code)


def mouse_up_left():
    _run(["click", "--", "0x80"])  # release-only (ydotool button-up code)


def scroll(amount: int):
    """Positive = scroll up, negative = scroll down. Amount is arbitrary units; tune per feel."""
    # ydotool exposes scroll via key input on some setups; falling back to wheel via `click`
    # is not standard, so on most distros you'd map this to arrow/page keys instead:
    if amount > 0:
        _run(["key", "Up"])
    elif amount < 0:
        _run(["key", "Down"])


def key_press(key_name: str):
    """e.g. 'Left', 'Right', 'Page_Up', 'Page_Down', 'ctrl+plus', 'ctrl+minus'"""
    _run(["key", key_name])


def zoom_in():
    _run(["key", "ctrl+plus"])


def zoom_out():
    _run(["key", "ctrl+minus"])


def swipe_next():
    _run(["key", "Right"])


def swipe_prev():
    _run(["key", "Left"])


if __name__ == "__main__":
    print("Testing os_control — cursor should move to (500, 500) then click.")
    move_cursor_absolute(500, 500)
    click_left()
