'''
利用模糊集合进行空间滤波
gg
'''
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math

# 输入（灰度差）模糊化函数
def fuzzy_knowledgement_filter(gray_difference):
    # 灰度归一化
    # grayscale = float('%.2f'%(GRAYSCALE/255))
    # 灰度差的隶属度函数为高斯函数一部分，参数可调
    # for x in range(0,2):
    #     for y in range(0,2):
    #         if (gray_difference[x][y]<=0.2)&(gray_difference[x][y]>=-0.2):
    #             # gray_difference[x][y] = math.exp(-20*(gray_difference[x][y])**2)
    #             gray_difference_w[x][y]= 1-abs(gray_difference[x][y])*5
    #         else:
    #             gray_difference_w[x][y]= 0
    if (gray_difference[0][1]<=0.2)&(gray_difference[0][1]>=-0.2):
        d2 = math.exp(-40 * gray_difference[0][1] * gray_difference[0][1])
    else:d2 = 0
    if (gray_difference[1][2]<=0.2)&(gray_difference[1][2]>=-0.2):
        d6 = math.exp(-40 * gray_difference[1][2] * gray_difference[1][2])
    else:d6 = 0
    if (gray_difference[2][1]<=0.2)&(gray_difference[2][1]>=-0.2):
        d8 = math.exp(-40 * gray_difference[2][1] * gray_difference[2][1])
    else:d8 = 0
    if (gray_difference[1][0]<=0.2)&(gray_difference[1][0]>=-0.2):
        d4 = math.exp(-40 * gray_difference[1][0] * gray_difference[1][0])
    else:d4 = 0

    # 运用模糊规则得到模糊化结果
    w1 = float('%.2f'%(min(d2,d6)))
    w2 = float('%.2f'%(min(d6,d8)))
    w3 = float('%.2f'%(min(d8,d4)))
    w4 = float('%.2f'%(min(d4,d2)))
    b = float('%.2f'%(min(1-w1,1-w2,1-w3,1-w4)))
    w = max(w1, w2, w3, w4)

    return w1,w2,w3,w4,b,w

# 输出模糊化函数
def output_fuzzification(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # 原图补一圈 0
    img_border1 = np.zeros([img.shape[0]+2,img.shape[1]+2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_border1[i+1,j+1] = img[i,j]/255.0
    # 计算各像素4领域的灰度差
    z = np.zeros([3,3])
    for y in range(2,img.shape[0]+1):
        for x in range(2,img.shape[1]+1):
            # z[0,0] = img_border1[y-1,x-1]-img_border1[y,x]
            z[0,1] = img_border1[y+1,x]-img_border1[y,x]
            # z[0,2] = img_border1[y-1,x+1]-img_border1[y,x]
            z[1,0] = img_border1[y,x-1]-img_border1[y,x]
            z[1,2] = img_border1[y,x+1]-img_border1[y,x]
            # z[2,0] = img_border1[y+1,x-1]-img_border1[y,x]
            z[2,1] = img_border1[y-1,x]-img_border1[y,x]
            # z[2,2] = img_border1[y+1,x+1]-img_border1[y,x]

            # 计算输入的隶属度函数
            w1,w2,w3,w4,b,w = fuzzy_knowledgement_filter(z)
            print(w1,w2,w3,w4,b)
            # 裁剪输出的隶属度函数
            v1 = float('%.2f'%(0.8*w1 + 0.2))
            v2 = float('%.2f'%(0.8*w2 + 0.2))
            v3 = float('%.2f'%(0.8*w3 + 0.2))
            v4 = float('%.2f'%(0.8*w4 + 0.2))
            v5 = float('%.2f'%(0.8-0.8*b))

            # 重心法反模糊化，计算输出
            g = np.zeros([img.shape[1]+2,img.shape[0]+2])
            # g[y,x] = (w1*v1 + w2*v2 + w3*v3 + w4*v4 + b*v5)/(w1+w2+w3+w4+b)

            data = 0
            MQ = 0
            for i in range(256):
                Q = max(min(i+1, b), min(i+1, w))
                data += i * Q
                MQ += Q
                data /= MQ
                g[y,x] = data

    return g

if __name__ == '__main__':
    img_gray = cv.imread('image_2\\gray\\512\\barbara.bmp')
    g = output_fuzzification(img_gray)
    plt.subplot(121),plt.imshow(img_gray,'gray'),plt.title('original')
    plt.subplot(122),plt.imshow(g,'gray'),plt.title('fuzzy_filter')
    plt.show()





