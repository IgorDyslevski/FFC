from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6 import QtWidgets
from pyui.mainscreen import Ui_Form
import sys
import cv2
from ApplicationResource.model import Witch, IncorrectModeCompareError, IncorrectSettingError, CameraPositionError

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
        self.mag = Witch()
        self.mag.set_for_searching_by_yolo()
        self.recording = False
        self.recorded = False
        self.big = True
        self.videowritter = None

    def modeon0(self):
        global mode, path_video, path_result_video
        mode = 0
        path_video = self.getFileFrom()
        self.path_result_video = self.getFileTo()
        self.resolution = self.getint('Exited resolution (square)')
        self.camera = cv2.VideoCapture(path_video)
        self.videowritter = cv2.VideoWriter(self.path_result_video,
                                            cv2.VideoWriter_fourcc(*'h264'), self.camera.get(cv2.CAP_PROP_FPS),
                                            (self.resolution, self.resolution))
        while True:
            face = self.get_frame_video()
            try:
                cv2.imshow('face', face)
            except cv2.error:
                print('Video ended')
                exit(0)

    def modeon1(self):
        global mode, path_result_video
        mode = 1
        self.path_result_video = self.getFileTo()
        self.resolution = self.getint('Exited resolution (square)')
        self.camera = cv2.VideoCapture(0)
        self.videowritter = cv2.VideoWriter(self.path_result_video,
                                            cv2.VideoWriter_fourcc(*'h264'), 60,
                                            (self.resolution, self.resolution))
        while True:
            face = self.get_frame_video()
            cv2.imshow('face', face)

    def gettext(self, text):
        text, okPressed = QtWidgets.QInputDialog.getText(self, "Get text", text, QtWidgets.QLineEdit.Normal, "")
        return text

    def getint(self, text):
        text, okPressed = QtWidgets.QInputDialog.getInt(self, "Get integer", text)
        return text

    def getFileTo(self):
        return QtWidgets.QFileDialog.getSaveFileName(self, 'Выберите, куда сохранить файлы')[0]

    def getFileFrom(self):
        return QtWidgets.QFileDialog.getOpenFileName(self, 'Выберите файл')[0]

    def get_frame_video(self):
        camera, big = self.camera, self.big
        mag = self.mag
        # getting ret and frame from camera
        ret, frame = camera.read()
        # getting face coordinates from camera
        try:
            faces_coords = mag.search_faces_from_raw_photo_yolo(frame)
        except Exception as e:
            if self.recording:
                self.videowritter.release()
                print('saved')
                exit(0)
            return 'error'
        copy_frame = frame.copy()
        main_face = None
        # putting index and rectangle into frame
        for i in range(1, len(faces_coords) + 1):
            x, y, w, h = faces_coords[i - 1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        if mag.last_face_finded:
            if faces_coords.__len__() > 0:
                if mag.camera_coords is None:
                    mag.set_camera_coords((*mag.get_mean_coords(mag.last_face_coords),))
                height, width, channels = copy_frame.shape
                x, y, _, _, index = mag.move_camera_to_compare_face('d', faces_coords, copy_frame)
                x, y = mag.make_stable((x, y), (600, 600), (width, height))
                main_face = copy_frame[y - 300:y + 300, x - 300:x + 300]
                # mag.last_data = x, y, w, h
                main_face = cv2.resize(main_face, (height, height))
            else:
                mag.unset_last_face()
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
                    elif key == 's':
                        print('Recording has started')
                        self.recording = True
                        self.recorded = True
                    elif key == 'e' and self.recording:
                        print('Recording has ended')
                        self.recording = False
                    elif key == 'q':
                        exit(0)
            except ValueError:
                print('Please, type a number')
        if main_face is None:
            rframe = frame
        else:
            rframe = main_face
        h, w, _ = rframe.shape
        rframe = mag.make_borders(rframe)
        rframe = cv2.resize(rframe, (rframe.shape[0], rframe.shape[0]))
        if self.recording:
            self.videowritter.write(cv2.resize(rframe, (self.resolution, self.resolution)))
            print('saving')
        elif self.recorded and not self.recording:
            self.videowritter.release()
            print('saved')
            exit(0)
        return rframe


apl = QApplication(sys.argv)
app = App()
app.show()
sys.exit(apl.exec())