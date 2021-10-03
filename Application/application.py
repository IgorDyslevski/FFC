from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6 import QtWidgets
from pyui.mainscreen import Ui_Form
import sys

# режим работы программы (0 - с записанным видео, 1 - online)
mode = None
# путь к видео (указывается, если mode = 0)
path_video = None
class App(Ui_Form, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.workswithvideobutton.clicked.connect(self.modeon0)
        self.recordvideowithprocess.clicked.connect(self.modeon1)

    def modeon0(self):
        global mode, path_video
        mode = 0
        path_video = QtWidgets.QFileDialog.getOpenFileName(self, 'Выберите файл')[0]

    def modeon1(self):
        global mode
        mode = 1


apl = QApplication(sys.argv)
app = App()
app.show()
sys.exit(apl.exec())