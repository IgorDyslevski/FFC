from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6 import QtWidgets
from pyui.mainscreen import Ui_Form
import sys

class App(Ui_Form, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

apl = QApplication(sys.argv)
app = App()
app.show()
sys.exit(apl.exec())
