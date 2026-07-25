import os
import cv2
from img_loader import load_img, im_info
from preprocess import resize_img, bgr_gr, gaussian_blur



def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    im_path = os.path.join(BASE_DIR, "datasets", "normal", "im1.jpg")
    im = load_img(im_path)

    if im is None:
        return
    im=resize_img(im)
    im_gr=bgr_gr(im)
    im_blur=gaussian_blur(im_gr)
    im_info(im)
    cv2.imshow("Bogie Vision Studio(BGR)", im)
    cv2.imshow("Bogie Vision Studio(GrayScale)", im_gr)
    cv2.imshow("Bogie Vision Studio(Blurred)", im_blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()