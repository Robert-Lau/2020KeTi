import cv2 as cv
import json
import retinex
import numpy

img = cv.imread("G:\\2020KeTi\\basic-learning\\floor.jpg")
with open('config.json', 'r') as f:
    config = json.load(f)

img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
cv.imshow("image",img)
cv.imshow("msrcr",img_msrcr)
cv.waitKey(0)