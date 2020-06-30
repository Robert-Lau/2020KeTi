# 利用模糊集合进行灰度变换和空间滤波
# 实现过程中遇到很多问题
'''
问题1：灰度值归一化，保留两位小数，很基础但是很重要
问题2：反模糊化过程中，结果的输出需要反归一化，乘以255
问题3：执行遍历图片像素时，从height到width，图片像素访问格式为：img[height,width]
                                        图片大小获取格式为：height,width,channel = img.shape()
'''
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img_gray = cv.imread("image\\lena.bmp",cv.IMREAD_GRAYSCALE)

def fuzzification(GRAYSCALE):
    # 灰度归一化
    grayscale = float("%.2f"%(GRAYSCALE/255))
    ### 模糊规则：
    '''
    R1：IF 一个像素是暗的，THEN 让这个像素更暗；
    R2：IF 一个像素是灰的，THEN 让他保持是灰的；
    R3：IF 一个像素是亮的，THEN 让这个像素更亮；
    '''
    ### 输入的隶属度函数
    if grayscale <= 0.27:
        dark = 1
    elif grayscale >= 0.5:
        dark = 0
    else:
        dark = float('%.2f'%((0.5-grayscale)/0.22))

    if grayscale >= 0.72:
        brig = 1
    elif grayscale <= 0.5:
        brig = 0
    else:
        brig = float("%.2f"%((grayscale - 0.5)/0.22))

    if grayscale <= 0.27:
        gray = 0
    elif grayscale >= 0.72:
        gray = 0
    else:
        gray = float('%.2f'%((0.72-grayscale)/0.22))

    return dark,gray,brig


def defuzzification(img):
    '''
    反模糊化函数
    :param img: 图片路径
    :return: 利用模糊规则灰度变换的结果
    '''
    # img_gray = cv.imread(img,cv.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    # 问题3
    g = np.zeros([height,width])
    for i in range(height):
        for j in range(width):
            dark, gray, brig = fuzzification(img[i,j])
            print(dark,gray,brig)
            g[i,j] = ((dark * 0) + (gray * 127) + (brig * 255))/(dark + gray + brig) # 注意反归一化
    return g

if __name__ == '__main__':
    test_fuzzy = defuzzification(img_gray)
    plt.subplot(221),plt.imshow(img_gray,'gray'),plt.title("origin")
    plt.subplot(222),plt.imshow(test_fuzzy,'gray'),plt.title("fuzzy")
    plt.subplot(223),plt.hist(img_gray.ravel(),256,[0,256]),plt.title("origin_his")
    plt.subplot(224), plt.hist(test_fuzzy.ravel(), 256, [0, 256]),plt.title('fuzzy_his')
    plt.show()
    # cv.imshow("1",img_gray)
    # cv.waitKey(-1)




