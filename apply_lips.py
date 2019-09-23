import face_detect as fd
import sys
import numpy as np
import cv2
from scipy.interpolate import interp1d

def get_points_lips(face_shape):
    """ Get the points for the lips. """
    ut = []
    ub =[]
    lt =[]
    lb = []
    upper_lips = []
    lower_lips = []

    # UpperLips
    ut.extend(face_shape[48:55])
    ub.extend(face_shape[48:49])
    ub.extend(face_shape[60:65])
    ub.extend(face_shape[54:55])

    # LowerLips
    lt.extend(face_shape[48:49])
    lt.extend(face_shape[54:55])
    lt.extend(face_shape[60:61])
    lt.extend(face_shape[64:68])
    lb.extend(face_shape[48:49])
    lb.extend(face_shape[54:60])

    ut = np.array(ut)
    ub = np.array(ub)
    lt = np.array(lt)
    lb = np.array(lb)

    f_ut = interp1d(ut[:,0], ut[:,1], kind="cubic")
    f_ub = interp1d(ub[:,0], ub[:,1], kind="cubic")
    f_lt = interp1d(lt[:,0], lt[:,1], kind="cubic")
    f_lb = interp1d(lb[:,0], lb[:,1], kind="cubic")

    x_ut = np.arange(min(ut[:,0]), max(ut[:,0]), 1)
    x_ub = np.arange(min(ub[:, 0]), max(ub[:, 0]), 1)
    x_lt = np.arange(min(lt[:, 0]), max(lt[:, 0]), 1)
    x_lb = np.arange(min(lb[:, 0]), max(lb[:, 0]), 1)

    ut = np.rot90(np.concatenate(([f_ut(x_ut)], [x_ut]), axis=0), 3).astype("int")
    ub = np.rot90(np.concatenate(([f_ub(x_ub)], [x_ub]), axis=0),3).astype("int")
    lt = np.rot90(np.concatenate(([f_lt(x_lt)], [x_lt]), axis=0), 3).astype("int")
    lb = np.rot90(np.concatenate(([f_lb(x_lb)], [x_lb]), axis=0),3).astype("int")

    upper_lips.extend(ut)
    upper_lips.extend(ub)

    lower_lips.extend(lt)
    lower_lips.extend(lb)

    upper_lips = np.array(upper_lips)
    lower_lips = np.array(lower_lips)

    return upper_lips, lower_lips


def contours_lips(image, upper_points, lower_points):
    cimg = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for (x, y) in upper_points:
        cv2.circle(cimg, (x, y), 1, 255, -1)

    for (x, y) in lower_points:
        cv2.circle(cimg, (x, y), 1, 255, -1)

    black = cimg
    mask_base = cv2.copyMakeBorder(cimg, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    x_cor = int(sum(upper_points[:,0])/upper_points.shape[0])
    y_cor = int(sum(upper_points[:,1])/upper_points.shape[0])
    cv2.floodFill(black, mask_base,(x_cor, y_cor) , 255)

    x_cor = int(sum(lower_points[:, 0]) / lower_points.shape[0])
    y_cor = int(sum(lower_points[:, 1]) / lower_points.shape[0])
    cv2.floodFill(black, mask_base, (x_cor, y_cor), 255)

    mask = np.array(np.where(cimg == 255, bool(True), bool(False)))

    return mask



def fill_lips(image, mask, color_bgr):
    overlay = np.full((image.shape[0], image.shape[1], 4), color_bgr, dtype='uint8')
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2BGRA)
    image[mask] = cv2.addWeighted(overlay[mask], 0.7, image[mask].copy(), 0.4, 0)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGRA2BGR)

    return image

def apply_lips(mask, image_after, color_bgr):
    mask_kari = np.where(mask == True, 255, 0).astype("uint8")
    mask_kari = cv2.cvtColor(mask_kari, cv2.COLOR_GRAY2BGR)
    mask_kari = cv2.GaussianBlur(mask_kari, (3, 3), 0)
    mask_kari = cv2.cvtColor(mask_kari, cv2.COLOR_BGR2GRAY)
    mask_big = np.where(mask_kari == 0, 0, 255).astype("uint8")
    mask_small = np.where(mask_kari == 255, 255, 0).astype("uint8")

    mask_big = cv2.cvtColor(mask_big, cv2.COLOR_GRAY2BGR)
    mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

    # Apply lips
    image_after = fill_lips(image_after.copy(), mask, color_bgr)
    image_after = np.where((mask_big == [255, 255, 255]) & (mask_small == [0, 0, 0]),
                           cv2.blur(image_after.copy(), (3, 3), 0), image_after.copy())

    return image_after

def main():
    # Get title
    title = sys.argv[2].split('\\')[-1].split('.')[0]

    face_data = fd.face_detect(sys.argv[1], sys.argv[2], title)
    image = cv2.imread(sys.argv[2])
    image_after = image.copy()
    color_bgr = tuple(sys.argv[3:7])

    for item in face_data:
        face_shape = item[2]
        upper_lips, lower_lips = get_points_lips(face_shape)

        mask = contours_lips(image.copy(), upper_lips, lower_lips)

        image_after = apply_lips(mask, image_after, color_bgr)

    # Save pic
    cv2.imwrite("{}_executed.jpg".format(title), image_after)

if __name__ == '__main__':
    main()