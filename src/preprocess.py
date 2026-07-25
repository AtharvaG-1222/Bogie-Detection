import cv2

def resize_img(image, width=800, height=600):
    resized = cv2.resize(image, (width, height))
    return resized

def bgr_gr(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    return gray

def gaussian_blur(im):
    blurred=cv2.GaussianBlur(im,(5,5),0)
    return blurred

def canny(im):
    edges=cv2.Canny(im, 50, 150)
    return edges