import os
import platform
import sys

# Preload torch before PyQt on Windows to avoid c10.dll WinError 1114
if platform.system() == "Windows":
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("Torch preload warning:", e)

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())