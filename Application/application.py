from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6 import QtWidgets
from pyui.mainscreen import Ui_Form
import sys
import cv2
from ApplicationResource.model import Magician

# режим работы программы (0 - с записанным видео, 1 - online)
mode = None
# путь к видео (указывается, если mode = 0)
path_video = None
# путь к последующему видео
path_result_video = None
class App(Ui_Form, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.workswithvideobutton.clicked.connect(self.modeon0)
        self.recordvideowithprocess.clicked.connect(self.modeon1)

        self.camera = None
        self.mag = Magician('ApplicationResource/haarcascades/haarcascade_frontalface_default.xml')
        self.big = True

    def modeon0(self):
        global mode, path_video, path_result_video
        mode = 0
        path_video = QtWidgets.QFileDialog.getOpenFileName(self, 'Выберите файл')[0]
        path_result_video = QtWidgets.QFileDialog.getSaveFileName(self, 'Выберите, куда сохранить файлы')[0]
        self.camera = cv2.VideoCapture(path_video)
        while True:
            cv2.imshow('face', self.get_frame_video())

    def modeon1(self):
        global mode, path_result_video
        mode = 1
        path_result_video = QtWidgets.QFileDialog.getSaveFileName(self, 'Выберите, куда сохранить файлы')[0]

    def gettext(self, text):
        text, okPressed = QtWidgets.QInputDialog.getText(self, "Get text", text, QtWidgets.QLineEdit.Normal, "")
        return text

    def get_frame_video(self):
        camera, big = self.camera, self.big
        mag = self.mag
        # getting ret and frame from camera
        ret, frame = camera.read()
        # getting face coordinates from camera
        faces_coords = mag.searhc_faces_from_raw_photo_yolo(frame)
        copy_frame = frame.copy()
        main_face = None
        # putting index and rectangle into frame
        for i in range(1, len(faces_coords) + 1):
            x, y, w, h = faces_coords[i - 1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        if mag.last_face_finded == True:
            if faces_coords.__len__() > 0:
                index = mag.search_face_like_last_face_distance(faces_coords, copy_frame)
                # chack for correct faces are
                if index != -1:
                    mag.set_last_face(faces_coords[index], copy_frame)
                    dpw = 100
                    dph = 100
                    height, width, channels = copy_frame.shape
                    x, y, w, h = faces_coords[index]
                    xg, yg, wg, hg = mag.borders_coords(x, y, w, h, dpw, dph, width, height)
                    main_face = copy_frame[yg:hg, xg:wg]
                    main_face = cv2.resize(main_face, (height, height))
                else:
                    mag.unset_last_face()
            else:
                main_face = frame
        else:
            if big:
                main_face = copy_frame
        # displaying the frame
        cv2.imshow('frame', frame)
        # check for keyboard event
        key_keyboard = cv2.waitKey(27)
        try:
            key = chr(key_keyboard)
        except ValueError:
            key = -1
        if key != -1:
            try:
                try:
                    print(key)
                    mag.set_last_face(faces_coords[int(key) - 1], copy_frame)
                except ValueError:
                    if key == ' ':
                        mag.unset_last_face()
                        big = True
            except ValueError:
                print('Please, type a number')
        if main_face is None:
            rframe = frame
        else:
            rframe = main_face
        h, w, _ = rframe.shape
        if w != h:
            if w > h:
                rframe = cv2.copyMakeBorder(rframe, (w - h) // 2, (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT, None,
                                            value=0)
            else:
                rframe = cv2.copyMakeBorder(rframe, 0, 0, (h - w) // 2, (h - w) // 2, cv2.BORDER_CONSTANT, None,
                                            value=0)
        rframe = cv2.resize(rframe, (rframe.shape[0], rframe.shape[0]))
        return rframe


apl = QApplication(sys.argv)
app = App()
app.show()
sys.exit(apl.exec())