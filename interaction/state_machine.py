import time


class GestureStateMachine:
    def __init__(
        self,
        stable_frames_required: int = 4,
        exit_frames_required: int = 6,
        confidence_threshold: float = 0.85,
        cooldown_sec: float = 0.3,
    ):
       
        self.stable_frames_required = stable_frames_required
        self.exit_frames_required = exit_frames_required
        self.confidence_threshold = confidence_threshold
        self.cooldown_sec = cooldown_sec

        self.state = "IDLE"
        self._consecutive_matches = 0
        self._exit_count = 0  # counts consecutive unconfident frames in ACTIVE
        self._cooldown_start = 0.0
        self._entered_active_this_frame = False
        self._released_this_frame = False

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, gesture_active: bool, confidence: float) -> str:
        """
        Call once per frame.
        gesture_active  True when the gesture of interest is the top prediction.
        confidence      Softmax probability for that prediction.
        Returns the current state string.
        """
        self._entered_active_this_frame = False
        self._released_this_frame = False

        if self.state == "COOLDOWN":
            if time.time() - self._cooldown_start >= self.cooldown_sec:
                self.state = "IDLE"
            return self.state

        confident = gesture_active and confidence >= self.confidence_threshold

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self.state == "IDLE":
            if confident:
                self.state = "DETECTING"
                self._consecutive_matches = 1
            return self.state

        # ── DETECTING ─────────────────────────────────────────────────────────
        if self.state == "DETECTING":
            if confident:
                self._consecutive_matches += 1
                if self._consecutive_matches >= self.stable_frames_required:
                    self.state = "CONFIRMED"
            else:
                # Entry is strict: any break resets the counter.
                self.state = "IDLE"
                self._consecutive_matches = 0
            return self.state

        # ── CONFIRMED ─────────────────────────────────────────────────────────
        if self.state == "CONFIRMED":
            self.state = "ACTIVE"
            self._entered_active_this_frame = True
            self._exit_count = 0
            return self.state

        # ── ACTIVE ────────────────────────────────────────────────────────────
        if self.state == "ACTIVE":
            if not confident:
                self._exit_count += 1
                if self._exit_count >= self.exit_frames_required:
                    # Sustained loss of confidence — intentional release.
                    self._exit_count = 0
                    self.state = "RELEASED"
                    self._released_this_frame = True
                # else: brief hiccup — stay ACTIVE, give it another frame.
            else:
                self._exit_count = 0  # confidence recovered, reset the countdown
            return self.state

        # ── RELEASED ──────────────────────────────────────────────────────────
        if self.state == "RELEASED":
            self.state = "COOLDOWN"
            self._cooldown_start = time.time()
            return self.state

        return self.state

    # ── query helpers ─────────────────────────────────────────────────────────

    def just_activated(self) -> bool:
        """True for exactly the one frame when ACTIVE is first entered."""
        return self._entered_active_this_frame

    def just_released(self) -> bool:
        """True for exactly the one frame when ACTIVE→RELEASED transition fires."""
        return self._released_this_frame

    def is_held(self) -> bool:
        """True every frame the gesture is continuously active."""
        return self.state == "ACTIVE"
