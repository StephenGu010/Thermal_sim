"""Tiny1-C thermal scope PC viewer entry point."""
from __future__ import annotations

import sys
import traceback

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Tiny1-C Thermal Scope PC Viewer")
    window = MainWindow()

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        window.append_log(f"[uncaught] {msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

##test
