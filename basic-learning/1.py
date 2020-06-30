import cv2 as cv
import numpy as np

img = cv.imread('zhuanzaiji.jpg')

min_nonzero = min(img[np.nonzero(img)])
img[img==0] = min_nonzero
print(min_nonzero)