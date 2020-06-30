# 非锐化掩蔽
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


src = cv.imread("image\\lenacolor.png")
src = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# 模糊
img = cv.medianBlur(src,11)
mask = cv.bitwise_not(cv.bitwise_and(src,cv.bitwise_not(img)))

changed = cv.bitwise_and(src,src,mask=7*mask)

cv.imshow('1',src)
cv.imshow('2',img)
cv.imshow('3',mask)
cv.imshow('4',changed)
cv.waitKey(-1)
