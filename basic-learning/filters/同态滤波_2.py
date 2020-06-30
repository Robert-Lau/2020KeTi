import os
import cv2
from PIL import Image
import numpy as np

'''
未转到对数域，直接傅里叶变换
'''

def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst



path = "G:/2020KeTi/basic-learning/qianguaban.jpg"
if os.path.isfile(path):
    print("path {} is existence;".format(path))
    img = Image.open(path)
    Img = img.convert('L')
    img = np.array(img)
    print(img, img.shape)

    image = cv2.imread(path)
    b,g,r = cv2.split(img)
    img_new_b = homomorphic_filter(b)
    img_new_g = homomorphic_filter(g)
    img_new_r = homomorphic_filter(r)
    img_new = cv2.merge([r,g,b])

    print("new img shape is {}".format(img_new.shape))
    img_new_gamma = np.power(img_new/255,0.5)
    cv2.imshow('original image',image)
    cv2.imshow("filtered image", img_new)
    cv2.imshow("filtered image gamma", img_new_gamma)
    cv2.waitKey(-1)
