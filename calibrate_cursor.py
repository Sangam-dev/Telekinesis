import sys
import subprocess
import time

try:
    from PyQt5.QtCore import Qt, QTimer, QRect
    from PyQt5.QtGui import QColor, QPainter, QPen, QFont
    from PyQt5.QtWidgets import QApplication, QWidget
except ImportError:
    print("Error: PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)


class CalibrationWindow(QWidget):
    """Shows crosshairs at calculated cursor position and actual OS cursor position."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cursor Calibration — Hold POINT gesture steady")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QRect(0, 0, 1920, 1200))
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(32)  # ~30 FPS
        
        self._orb_x = 960
        self._orb_y = 540
        self._os_x = 960
        self._os_y = 540
        
        self._measurements = []
        self._measuring = False

    def _on_tick(self):
        """Read OS cursor position and redraw."""
        try:
            import subprocess
            out = subprocess.check_output(
                ["xdotool", "getmouselocation"],
                timeout=0.5,
                stderr=subprocess.DEVNULL
            ).decode()
            self._os_x = int(out.split("x:")[1].split()[0])
            self._os_y = int(out.split("y:")[1].split()[0])
        except Exception:
            pass
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        
        # Draw large crosshair at OS cursor position (RED = actual system cursor)
        p.setPen(QPen(QColor(255, 0, 0, 200), 3))
        p.drawLine(self._os_x - 40, self._os_y, self._os_x + 40, self._os_y)
        p.drawLine(self._os_x, self._os_y - 40, self._os_x, self._os_y + 40)
        
        # Draw large circle around it
        p.setPen(QPen(QColor(255, 0, 0, 150), 2))
        p.drawEllipse(self._os_x - 50, self._os_y - 50, 100, 100)
        
        # Draw text label
        p.setPen(QColor(255, 0, 0, 220))
        font = QFont("Monospace", 12)
        p.setFont(font)
        p.drawText(self._os_x + 60, self._os_y - 10, f"OS Cursor ({self._os_x}, {self._os_y})")
        
        # Draw text instructions
        p.setPen(QColor(255, 255, 255, 220))
        font = QFont("Monospace", 14, QFont.Bold)
        p.setFont(font)
        p.drawText(50, 100, "RED = OS CURSOR position  |  BLUE = ORB position")
        p.drawText(50, 130, "Hold POINT gesture perfectly still for calibration")
        p.drawText(50, 160, "Measure the pixel offset between the two")
        p.drawText(50, 190, "Press Ctrl+C to exit")
        
        p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


def main():
    print("=== Cursor Calibration Tool ===")
    print()
    print("Instructions:")
    print("1. Start Telekinesis (python3 main.py)")
    print("2. Activate POINT gesture and hold your finger perfectly still")
    print("3. A window will show RED crosshair (OS cursor) and instructions")
    print("4. Measure the pixel offset between where your orb is and the RED crosshair")
    print("5. Update config/cursor_calibration.py with the offsets")
    print()
    print("If RED crosshair is 100 pixels to the RIGHT of your orb:")
    print("  → Set CURSOR_OFFSET_X = -100")
    print()
    print("If RED crosshair is 50 pixels DOWN from your orb:")
    print("  → Set CURSOR_OFFSET_Y = -50")
    print()
    print("Starting calibration window...")
    
    app = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    window.show()
    
    print("\nCalibration window opened. Press Ctrl+C or close window to exit.")
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nCalibration exited.")
        sys.exit(0)


if __name__ == "__main__":
    main()
