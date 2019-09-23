from imutils import face_utils
import numpy as np
import dlib
import cv2

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def face_detect(dat_path, img_path, title):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dat_path)

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    face_data = []

    for (i, face) in enumerate(faces):

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(image, "Face No.{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        cv2.imwrite("{}_detected.jpg".format(title), image)

        face_data.append((i,face,shape))

    return face_data