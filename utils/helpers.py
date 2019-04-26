import jpeg4py as jpeg
import cv2


def load_img_fast_jpg(img_path):
    try:
        result = jpeg.JPEG(img_path).decode()
    except:
        result = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return result
