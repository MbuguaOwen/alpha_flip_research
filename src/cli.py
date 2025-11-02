import sys


def _safe_write(s: str):
    try:
        sys.stdout.write(s)
    except Exception:
        # Fallback for narrow consoles/encodings
        sys.stdout.write(s.encode("ascii", "replace").decode("ascii"))
    sys.stdout.flush()


def info(msg: str):
    print(f"[i] {msg}")


def ok(msg: str):
    print(f"[ok] {msg}")


def warn(msg: str):
    print(f"[!] {msg}")


def error(msg: str):
    print(f"[x] {msg}")


class ProgressBar:
    def __init__(self, total: int, prefix: str = "Progress", bar_len: int = 30):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.bar_len = bar_len
        self.completed = 0
        self._last_drawn = -1
        self._draw()

    def _draw(self):
        frac = min(max(self.completed / self.total, 0.0), 1.0)
        filled = int(self.bar_len * frac)
        bar = "#" * filled + "-" * (self.bar_len - filled)
        pct = int(frac * 100)
        line = f"\r[{self.prefix}] |{bar}| {pct:3d}% ({self.completed}/{self.total})"
        _safe_write(line)

    def update(self, completed: int):
        self.completed = min(max(int(completed), 0), self.total)
        step = max(self.total // 100, 1)
        if self.completed == self.total or self.completed % step == 0:
            self._draw()

    def advance(self, step: int = 1):
        self.update(self.completed + int(step))

    def finish(self):
        self.completed = self.total
        self._draw()
        _safe_write("\n")

