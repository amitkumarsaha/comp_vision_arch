from __future__ import annotations

import sys
from pathlib import Path

from src.ui import MenuApp


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def main() -> None:
    MenuApp(ROOT, PYTHON).run()


if __name__ == "__main__":
    main()
