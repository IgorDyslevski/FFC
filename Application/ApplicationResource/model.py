import cv2
import yoloface
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
import numpy as np
from typing import Optional


class IncorrectSettingError(Exception):
    pass


class CameraPositionError(Exception):
    pass


class IncorrectModeCompareError(Exception):
    pass


class Magician:
    def __init__(self):
        self.unset_last_face()

    # метод для указания последнего найденного лица
    def set_last_face(self, coords_x_y_w_h: tuple, photo: np.array) -> None:
        self.last_face_coords = coords_x_y_w_h[:4]
        self.last_face_photo = photo
        self.last_face_finded = True

    # метод для обнуления последнего найденного лица
    def unset_last_face(self) -> None:
        self.last_face_finded = False
        self.last_face_coords = None
        self.last_face_photo = None

    def set_for_searching_by_opencv(self, path: str) -> None:
        self.path = path
        self.processor = cv2.CascadeClassifier(self.path)

    def set_for_searching_by_yolo(self) -> None:
        self.processor = yoloface.face_analysis()

    def set_model(self, path: str) -> None:
        self.model = load_model(path)

    def compare_lastface_and_newface_hist(self, faces_coords_x_y_w_h: tuple, photo: np.array) -> int:
        results_of_getting_distance = []
        last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords[:4]
        last_face = cv2.resize(
            self.last_face_photo[last_face_y:last_face_y+last_face_h, last_face_x:last_face_x + last_face_w], (128, 128)
        )
        last_face_hist = cv2.calcHist([last_face], [0], None, [256],
                                      [0, 256])
        for x, y, w, h in faces_coords_x_y_w_h:
            if x > -1 and y > -1:
                face = cv2.resize(photo[y:y+h, x:x+w], (128, 128))
            else:
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                face = cv2.resize(photo[y:y + h, x:x + w], (128, 128))
            face_hist = cv2.calcHist([face], [0], None, [256],
                                      [0, 256])

            results_of_getting_distance.append(
                cv2.compareHist(face_hist, last_face_hist, cv2.HISTCMP_BHATTACHARYYA)
            )
        index_of_min_differnce = results_of_getting_distance.index(min(results_of_getting_distance))
        return index_of_min_differnce

    def compare_lastface_and_newface_model(self, faces_coords_x_y_w_h: tuple, photo: np.array) -> int:
        results_of_getting_distance = []
        last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords[:4]
        last_face = cv2.resize(
            self.last_face_photo[last_face_y:last_face_y+last_face_h, last_face_x:last_face_x + last_face_w], (224, 224)
        )
        last_face_emb = self.model.predict(last_face[None])
        for x, y, w, h in faces_coords_x_y_w_h:
            if x > -1 and y > -1:
                face = cv2.resize(photo[y:y+h, x:x+w], (224, 224))
            else:
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                face = cv2.resize(photo[y:y + h, x:x + w], (224, 224))
            face_emb = self.model.predict(face[None])

            results_of_getting_distance.append(
                cosine(last_face_emb, face_emb)
            )
        index_of_min_differnce = results_of_getting_distance.index(min(results_of_getting_distance))
        return index_of_min_differnce

    def compare_lastface_and_newface_distance(self, faces_coords_x_y_w_h: tuple, photo: np.array) -> int:
        height, width, channels = photo.shape
        results_of_getting_distance = []
        last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords[:4]
        last_face_central_x, last_face_central_y = self.get_mean_coords((last_face_x, last_face_y,
                                                                         last_face_w, last_face_h))
        for x, y, w, h in faces_coords_x_y_w_h:
            central_x, central_y = x + w // 2, y + h // 2
            results_of_getting_distance.append(
                (abs(last_face_central_x - central_x) ** 2 + abs(last_face_central_y - central_y) ** 2) ** 0.5
            )
        index_of_min_differnce = results_of_getting_distance.index(min(results_of_getting_distance))
        return index_of_min_differnce

    def search_faces_from_raw_photo_cv2(self, photo: np.array) -> Optional[tuple]:
        try:
            gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
            faces_coords_x_y_w_h = self.processor.detectMultiScale(gray_photo, 1.2, 1)
            faces_coords_x_y_w_h = list(sorted(faces_coords_x_y_w_h, key=lambda x: x[2], reverse=True))
            return faces_coords_x_y_w_h
        except AttributeError:
            raise IncorrectSettingError('Please, set up for searching faces by opencv. '
                                   'Write Magician.set_for_searching_by_opencv(path_cascade)')

    def search_faces_from_raw_photo_yolo(self, photo: np.array) -> Optional[tuple]:
        try:
            _, box, conf = self.processor.face_detection(frame_arr=photo, frame_status=True, model='tiny')
            faces_coords_x_y_w_h = []
            if box.__len__() > 0:
                for item in box:
                    a = [item[0], item[1], item[3], item[2]]
                    faces_coords_x_y_w_h.append(a)
            faces_coords_x_y_w_h = list(sorted(faces_coords_x_y_w_h, key=lambda x: (x[0], x[1])))
            return faces_coords_x_y_w_h
        except AttributeError:
            raise IncorrectSettingError('Please, set up for searching faces by yolo. '
                                   'Write Magician.set_for_searching_by_yolo()')

    def get_mean_coords(self, coords_x_y_w_h: tuple) -> tuple:
        x, y, w, h = coords_x_y_w_h[:4]
        mean_x, mean_y = x + w // 2, y + h // 2
        return mean_x, mean_y


class Witch(Magician):
    def __init__(self):
        super().__init__()
        self.camera_coords = None
        self.searching_by = None

    def set_camera_coords(self, coords__mean_x__mean_y: tuple) -> None:
        self.camera_coords = coords__mean_x__mean_y[:2]

    def move_camera(self, coords__mean_x__mean_y: tuple) -> Optional[tuple]:
        if self.camera_coords is not None:
            coef = 10
            gx, gy = self.camera_coords[:2]
            x, y = coords__mean_x__mean_y[:2]
            if gx > x:
                gx -= (gx - x) // coef
            elif gx < x:
                gx += (-gx + x) // coef
            else:
                pass

            if gy > y:
                gy -= (gy - y) // coef
            elif gy < y:
                gy += (-gy + y) // coef
            else:
                pass
            self.camera_coords = (gx, gy)
            mean_x, mean_y = gx, gy
            return mean_x, mean_y
        else:
            raise CameraPositionError('Please^ set up camera coordinates before using')

    def move_camera_to_compare_face(self, mode: str, faces_coords_x_y_w_h: tuple, photo: np.array) -> tuple:
        if mode == 'd':
            index = self.compare_lastface_and_newface_distance(faces_coords_x_y_w_h, photo)
        elif mode == 'h':
            index = self.compare_lastface_and_newface_hist(faces_coords_x_y_w_h, photo)
        elif mode == 'm':
            index = self.compare_lastface_and_newface_model(faces_coords_x_y_w_h, photo)
        else:
            raise IncorrectModeCompareError('Mode can include only characters: m - model comparing, '
                                            'd - distance comparing, h - histogram comparing')
        x, y, w, h = faces_coords_x_y_w_h[index]
        self.set_last_face((x, y, w, h), photo)
        gx, gy = self.move_camera((*self.get_mean_coords((x, y, w, h)),))
        mean_x, mean_y = gx, gy
        return mean_x, mean_y, w, h, index

    def make_borders(self, frame: np.array) -> np.array:
        h, w, _ = frame.shape
        if w != h:
            if w > h:
                rframe = cv2.copyMakeBorder(frame, (w - h) // 2, (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT, None,
                                            value=0)
            else:
                rframe = cv2.copyMakeBorder(frame, 0, 0, (h - w) // 2, (h - w) // 2, cv2.BORDER_CONSTANT, None,
                                            value=0)
        else:
            rframe = frame
        return rframe

    def make_stable(self, coords__mean_x__mean_y: tuple, size_future_face: tuple, size_frame: tuple) -> tuple:
        x, y = coords__mean_x__mean_y[:2]
        w, h = size_future_face[:2]
        first_point = x - w // 2, y - h // 2
        second_point = x + w // 2, y + h // 2
        w, h = size_frame[:2]
        if first_point[0] < 0:
            x += 0 - first_point[0]
        if first_point[1] < 0:
            y += 0 - first_point[1]
        if second_point[0] > w:
            x -= second_point[0] - w
        if second_point[1] > h:
            y -= second_point[1] - h
        mean_x, mean_y = x, y
        return mean_x, mean_y
