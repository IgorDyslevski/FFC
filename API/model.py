import cv2
import yoloface
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine


class Magician:
    def __init__(self, path):
        self.path = path
        self.cascade = cv2.CascadeClassifier(self.path)
        self.last_face_finded = False
        self.last_face_coords = None
        self.last_face_photo = None
        self.face = yoloface.face_analysis()


        # self.model = load_model('model.h5')

    def set_last_face(self, coords, photo):
        self.last_face_coords = coords
        self.last_face_photo = photo
        self.last_face_finded = True

    def unset_last_face(self):
        self.last_face_finded = False
        self.last_face_coords = None
        self.last_face_photo = None

    def search_face_like_last_face_hist(self, faces_coords, photo):
        results_of_getting_distance = []
        last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords
        last_face = cv2.resize(
            self.last_face_photo[last_face_y:last_face_y+last_face_h, last_face_x:last_face_x + last_face_w], (128, 128)
        )
        last_face_hist = cv2.calcHist([last_face], [0], None, [256],
                                      [0, 256])
        for x, y, w, h in faces_coords:
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

    # def search_face_like_last_face_vgg16(self, faces_coords, photo):
    #     results_of_getting_distance = []
    #     last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords
    #     last_face = cv2.resize(
    #         self.last_face_photo[last_face_y:last_face_y+last_face_h, last_face_x:last_face_x + last_face_w], (224, 224)
    #     )
    #     last_face_emb = self.model.predict(last_face[None])
    #     for x, y, w, h in faces_coords:
    #         if x > -1 and y > -1:
    #             face = cv2.resize(photo[y:y+h, x:x+w], (224, 224))
    #         else:
    #             if x < 0:
    #                 x = 0
    #             if y < 0:
    #                 y = 0
    #             face = cv2.resize(photo[y:y + h, x:x + w], (224, 224))
    #         face_emb = self.model.predict(face[None])
    #
    #         results_of_getting_distance.append(
    #             cosine(last_face_emb, face_emb)
    #         )
    #     index_of_min_differnce = results_of_getting_distance.index(min(results_of_getting_distance))
    #     return index_of_min_differnce

    def search_face_like_last_face_distance(self, faces_coords, photo):
        height, width, channels = photo.shape
        results_of_getting_distance = []
        last_face_x, last_face_y, last_face_w, last_face_h = self.last_face_coords
        last_face_central_x, last_face_central_y = last_face_x + last_face_w // 2, last_face_y + last_face_h // 2
        for x, y, w, h in faces_coords:
            central_x, central_y = x + w // 2, y + h // 2
            results_of_getting_distance.append(
                (abs(last_face_central_x - central_x) ** 2 + abs(last_face_central_y - central_y) ** 2) ** 0.5
            )
        index_of_min_differnce = results_of_getting_distance.index(min(results_of_getting_distance))
        if results_of_getting_distance[index_of_min_differnce] > width // 5:
            return -1
        return index_of_min_differnce

    def search_faces_from_raw_photo_cv2(self, photo):
        gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        faces_coords = self.cascade.detectMultiScale(gray_photo, 1.2, 1)
        faces_coords = list(sorted(faces_coords, key=lambda x: (x[0], x[1])))
        return faces_coords

    def searhc_faces_from_raw_photo_yolo(self, photo):
        _, box, conf = self.face.face_detection(frame_arr=photo, frame_status=True, model='tiny')
        faces_coords = []
        if box.__len__() > 0:
            for item in box:
                a = [item[0], item[1], item[3], item[2]]
                faces_coords.append(a)
        faces_coords = list(sorted(faces_coords, key=lambda x: (x[0], x[1])))
        return faces_coords

    def borders_coords(self, x, y, w, h, dpw, dph, width, height):
        x_r = x + w + dpw - width - 1

        if w == h:
            pass
        elif w > h:
            dph += (w - h) // 2
        elif h > w:
            dpw += (h - w) // 2

        xg, yg, wg, hg = x, y, w, h
        x_r = x + w + dpw - width
        y_r = y + h + dph - height
        xr = 0
        if x - dpw < 0:
            xr = (x - dpw) * -1
        xl = 0
        if x + w + dpw - width > 0:
            xl = x + w + dpw - width
        yr = 0
        if y - dph < 0:
            yr = (y - dph) * -1
        yl = 0
        if y + h + dph - width > 0:
            yl = y + h + dph - height

        xg, yg, wg, hg = x - dpw + xr - xl, y - dph + yr - yl, x + w + dpw + xr - xl, y + h + dph + yr - yl
        return xg, yg, wg, hg