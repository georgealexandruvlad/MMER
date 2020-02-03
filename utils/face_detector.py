import cv2
from paths import XML_PATH
import os


def detect_face(img, face_area_size, clf):
    # only absolute path here!!

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(gray, 1.1, 24)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, face_area_size)
        return crop_img

    return None
