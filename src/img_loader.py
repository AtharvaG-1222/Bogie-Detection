import cv2

def load_img(image_path):

    im = cv2.imread(image_path)

    if im is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    return im

def im_info(im):
    height, width, channels=im.shape
    print("\n----- Image Information -----")
    print(f"Width    : {width} pixels")
    print(f"Height   : {height} pixels")
    print(f"Channels : {channels}")
    print(f"Data Type: {im.dtype}")
