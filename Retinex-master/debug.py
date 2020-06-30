import cv2 as cv
import json
import retinex
import numpy

img = cv.imread("3.jpg")
img_ssr = retinex.singleScaleRetinex(img,15)
# cv.imshow("image",img)
cv.imshow("ssr",img_ssr)
cv.waitKey(0)