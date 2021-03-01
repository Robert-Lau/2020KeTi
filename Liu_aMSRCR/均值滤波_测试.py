import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

src = cv.imread("H:\\2020KeTi\Liu_aMSRCR\\test\\19.jpg")
blur = cv.blur(src,(9,9))
high_freq_img = src - blur
cv.imshow("blur",blur)
cv.imshow("high_freq",high_freq_img)
cv.namedWindow("high_freq",cv.WINDOW_NORMAL)
cv.waitKey(-1)