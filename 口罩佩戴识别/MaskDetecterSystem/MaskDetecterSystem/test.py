###########################读取图片显示图片
import cv2 #导入opencv库

#读取一张图片，地址不能带中文
image=cv2.imread("images/1.jpg")

#显示图片，参数：（窗口标识字符串，imread读入的图像）
cv2.imshow("东小东标题",image)

#窗口等待任意键盘按键输入,0为一直等待,其他数字为毫秒数
cv2.waitKey(0)

#销毁窗口，退出程序
cv2.destroyAllWindows()


############################打开摄像头显示摄像头

import cv2 as cv

def video_demo():
#0是代表摄像头编号，只有一个的话默认为0
    capture=cv.VideoCapture(0) 
    while(True):
        ref,frame=capture.read()
 
        cv.imshow("1",frame)
#等待30ms显示图像，若过程中按“Esc”退出
        c= cv.waitKey(30) & 0xff 
        if c==27:
            capture.release()
            break
            
video_demo()
cv.waitKey()
cv.destroyAllWindows()

##############################分类器
import cv2

img = cv2.imread("images/1.jpg",1)#读取一张图片

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#将图片转化成灰度

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('img',img)
cv2.waitKey()

###############################人脸视频检测
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''

capture=cv2.VideoCapture(0) 
while(True):
    ref,frame=capture.read()
 
    #cv2.imshow("frame",frame)
    #等待30ms显示图像，若过程中按“Esc”退出
    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("face",frame)
cv.waitKey()
cv.destroyAllWindows()

############################人脸\眼睛\上半身 视频检测
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar眼镜特征分类器'''

upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
upperbody_cascade.load("data/haarcascades/haarcascade_upperbody.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar眼镜特征分类器'''


capture=cv2.VideoCapture(0) 
while(True):
    ref,frame=capture.read()
 
    #cv2.imshow("frame",frame)
    #等待30ms显示图像，若过程中按“Esc”退出
    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    upperbody = upperbody_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in eyes:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in upperbody:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("face",frame)
cv.waitKey()
cv.destroyAllWindows()


##############################霍夫变换加挑

import  cv2
import  numpy  as  np
def nothing(x):
    img  =  cv2.imread('images/21.jpg')
    gray  =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges  =  cv2.Canny(gray,50,150,apertureSize  =  3)
    minLineLength=cv2.getTrackbarPos("minLineLength", 'img')
    maxLineGap=cv2.getTrackbarPos("maxLineGap", 'img')
    lines  =  cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength,maxLineGap)
    for  x1,y1,x2,y2  in  lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow('img',img)
img  =  cv2.imread('images/21.jpg')
gray  =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
edges  =  cv2.Canny(gray,50,150,apertureSize  =  3)
cv2.imshow('edges',edges)
minLineLength  =  100
maxLineGap  =  50
cv2.imshow('img',img)
cv2.createTrackbar("minLineLength", 'img', 100, 500, nothing)
cv2.createTrackbar("maxLineGap", 'img', 50, 100, nothing)


while(0):
    minLineLength=cv2.getTrackbarPos("minLineLength", 'img')
    maxLineGap=cv2.getTrackbarPos("maxLineGap", 'img')
    lines  =  cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength,maxLineGap)
    for  x1,y1,x2,y2  in  lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

#####################hough
#15，霍夫直线检测
import cv2 as cv
import numpy as np
 
def line_detection(image):
    #将图像转换为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #利用Canny进行边缘检测
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    #设置霍夫直线检测参数
    #第二个参数为半径的步长，第三个参数为每次偏转的角度
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    print(type(lines))
    #将求得交点坐标反代换进行直线绘制
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.namedWindow("lines_demo",0)
    cv.imshow("lines_demo", image)
 
def line_detect_possible_demo(image):
    #通常用第二种方式。
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    #自动检测可能的直线，返回的是一条条线段
    minLineLength=cv2.getTrackbarPos("minLineLength", 'src')
    maxLineGap=cv2.getTrackbarPos("maxLineGap", 'src')
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength, maxLineGap)
    print(type(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("linepossible_demo", image)
def callback(x,src):
     line_detect_possible_demo(src)
src = cv.imread("images/21.jpg")
cv.namedWindow("src",0)
minLineLength  =  100
maxLineGap  =  50

cv2.createTrackbar("minLineLength", "src", 100, 500, callback(src))
cv2.createTrackbar("maxLineGap", "src", 50, 100, callback(src))
cv.imshow("src",src)

cv.waitKey(0)
cv.destroyAllWindows()

################################前景提取
import numpy as np
import cv2
 
cap = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    canny=cv2.Canny(fgmask,50,150)
    cv2.imshow('frame', canny)
    cv2.waitKey(30)
    # if cv2.waitKey(30) & 0xff:
    #     break
cap.release()
cv2.destroyAllWindows()
#######################################摄像头替换
# -*- coding: utf-8 -*-
"""
视频背景替换
"""
from PIL import Image
import numpy as np
import cv2
 
cap = cv2.VideoCapture(0)
cap.set(5,10)
 
# 要替换的背景
img_back=cv2.imread('img_back.jpg')
 
 
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    #获取图片的尺寸
    rows, cols, channels = frame.shape
 
    lower_color = np.array([120, 120, 120])
    upper_color = np.array([250, 250, 250])
    # 创建掩图
    fgmask = cv2.inRange(frame, lower_color, upper_color)
    cv2.imshow('Mask', fgmask)
 
    # 腐蚀膨胀
    erode = cv2.erode(fgmask, None, iterations=1)
    cv2.imshow('erode', erode)
    dilate = cv2.dilate(erode, None, iterations=1)
    cv2.imshow('dilate', dilate)
 
    rows, cols = dilate.shape
    img_back=img_back[0:rows,0:cols]
    print(img_back)
    # #根据掩图和原图进行抠图
    img2_fg = cv2.bitwise_and(img_back, img_back, mask=dilate)
    Mask_inv = cv2.bitwise_not(dilate)
    img3_fg = cv2.bitwise_and(frame, frame, mask=Mask_inv)
    finalImg=img2_fg+img3_fg
    cv2.imshow('res', finalImg)
 
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()


######vtest.avi检测
import numpy as np
import cv2
import time
import datetime

colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))#定义矩形颜色

#cap = cv2.VideoCapture(0) #参数为0是打开摄像头，文件名是打开视频
cap = cv2.VideoCapture("images/vtest.avi") #参数为0是打开摄像头，文件名是打开视频
fgbg = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法

fourcc = cv2.VideoWriter_fourcc(*'XVID')#设置保存图片格式
out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (768,576))#分辨率要和原视频对应


while True:
    ret, frame = cap.read()  #读取图片
    fgmask = fgbg.apply(frame)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪

    contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #寻找前景

    count=0
    for cont in contours:
        Area = cv2.contourArea(cont)  # 计算轮廓面积
        if Area < 300:  # 过滤面积小于10的形状
            continue

        count += 1  # 计数加一

        print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积

        rect = cv2.boundingRect(cont) #提取矩形坐标

        print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),colour[count%6],1)#原图上绘制矩形
        cv2.rectangle(fgmask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)  #黑白前景上绘制矩形

        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
        cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号



    cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) #显示总数
    cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    print("----------------------------")

    cv2.imshow('frame', frame)#在原图上标注
    cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景
    out.write(frame)
    k = cv2.waitKey(30)&0xff  #按esc退出
    if k == 27:
        break


out.release()#释放文件
cap.release()
cv2.destoryAllWindows()#关闭所有窗口

#######备份
import cv2
import time
import numpy as np

def line_detect_possible_demo(image):#霍夫变换找直线
        #通常用第二种方式。
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray1, 50, 150, apertureSize=3)
        #自动检测可能的直线，返回的是一条条线段
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
        print(type(lines))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("linepossible_demo", image)
def facesdetecter(image):
    start = time.time()#开始时间
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
    image=cv2.GaussianBlur(image,(5,5),0)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)#将图片转化成HSV格式
    #cv2.imshow("hsv",hsv)#显示HSV图
    H,S,V=cv2.split(hsv)
    #cv2.imshow("hsv-H",H)#显示HSV图明度
    #ret, thresh =cv2.threshold(H,17,25,0)#阈值分割
    thresh=cv2.inRange(H,30/2,50/2)#0-180du 提取人体肤色区域
    cv2.imshow("hsv-H-threshold",thresh)#显示二值化图

    canny=cv2.Canny(gray,50,150)
    #cv2.imshow("canny",canny)#显示边缘处理图

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#人脸检测
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)#眼睛检测
    upperbody = upperbody_cascade.detectMultiScale(gray, 1.3, 5)#上身检测
    mouth=mouth_cascade.detectMultiScale(gray, 1.3, 5)#嘴巴检测
    nose=nose_cascade.detectMultiScale(gray, 1.3, 5)#鼻子检测

    for (x,y,w,h) in faces:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in eyes:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in upperbody:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in mouth:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in nose:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    #计算帧率
    end = time.time()#结束时间
    fps=1 / (end-start)#帧率
    frame = cv2.putText(image,"fps:{:.3f}".format(fps),(3,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)#绘制
    cv2.imshow("face",image)#显示最终图片
    line_detect_possible_demo(image)#霍夫变换找直线

def mogseparate(image):
    fgmask = mog.apply(image)   
    ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("fgmask", fgmask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)    
    backgimage = mog.getBackgroundImage()
    #查找轮廓
    #contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    #cv2.drawContours(binary,contours,-1,(0,0,255),3)
    binary = cv2.erode(binary, None, iterations=4)# 腐蚀
    cv2.imshow('erode', binary)
    cv2.imshow("backgimage", backgimage)
    cv2.imshow("frame", image)
    cv2.imshow("binary", binary)
    
def knnseperate(image):
    mog_sub_mask = mog2_sub.apply(image)
    knn_sub_mask = knn_sub.apply(image)
 
    cv2.imshow('original', image)
    cv2.imshow('MOG2', mog_sub_mask)
    cv2.imshow('KNN', knn_sub_mask)
def trackAvg(image,k_write):
    if k_write==1:
        cv2.imwrite("images/backgound.jpg",image)
        k_write=0 
    background=cv2.imread("images/backgound.jpg")
    cv2.RunningAvg(image, background, 0.1, None)    
    cv2.imshow('live', frame)
    cv2.imshow('avg',avg_show)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar眼镜特征分类器'''

upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
upperbody_cascade.load("data/haarcascades/haarcascade_upperbody.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mouth_cascade.load("data/haarcascades/haarcascade_mcs_mouth.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
nose_cascade.load("data/haarcascades/haarcascade_mcs_nose.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mog = cv2.createBackgroundSubtractorMOG2()
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

k_write=1
capture=cv2.VideoCapture(0) 
while(True):
    
    ref,frame=capture.read()
    if ref==False:
        print("打开摄像头错误")
        break
    #cv2.imshow("frame",frame)
    #等待30ms显示图像，若过程中按“Esc”退出
    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
    #facesdetecter(frame)
    #mogseparate(frame)#mog方式分离前景
    #knnseperate(frame)#knn方式分离前景
    trackAvg(frame,k_write)#runningavg分离前景
    
cap.release()
cv2.destroyAllWindows()



#####################grabCut
# -*- coding:utf-8 -*-

import cv2
import numpy as np

# Step1. 加载图像
img = cv2.imread('images/messi5.jpg')

# Step2. 创建掩模、背景图和前景图
mask = np.zeros(img.shape[:2], np.uint8) # 创建大小相同的掩模
bgdModel = np.zeros((1,65), np.float64) # 创建背景图像
fgdModel = np.zeros((1,65), np.float64) # 创建前景图像

# Step3. 初始化矩形区域
# 这个矩形必须完全包含前景（相当于这里的梅西）
rect = (50,50,450,290)

# Step4. GrubCut算法，迭代5次
# mask的取值为0,1,2,3
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT) # 迭代5次

# Step5. mask中，值为2和0的统一转化为0, 1和3转化为1 
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:,:,np.newaxis] # np.newaxis 插入一个新维度，相当于将二维矩阵扩充为三维

cv2.imshow("dst", img)
cv2.waitKey(0)
###############################findContours
import cv2
img=cv2.imread('images/24.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
cv2.imshow("img",img)
cv2.waitKey(0)

###################################
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is debugged by Harden Qiu
"""

import cv2
import numpy as np

#肤色识别
def skin(frame):
    lower = np.array([0, 40, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    return skin

#求最大连通域的中心点坐标
def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

#主函数
def main():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame=cv2.flip(frame,1)
        #显示摄像头
        cv2.imshow('Original',frame)

        #皮肤粒子识别
        frame = skin(frame)

        #灰度
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #高斯滤波
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        #二值化
        ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

        #轮廓
        contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print("number of contours:%d" % len(contours))
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        #找到最大区域并填充
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        max_idx = np.argmax(area)
        for i in range(max_idx - 1):
            cv2.fillConvexPoly(frame, contours[max_idx - 1], 0)
        cv2.fillConvexPoly(frame, contours[max_idx], 255)

        #求最大连通域的中心坐标
        cnt_centroid = centroid(contours[max_idx])
        cv2.circle(contours[max_idx],cnt_centroid,5,[255,0,255],-1)
        print("Centroid : " + str(cnt_centroid))

        #处理后显示
        cv2.imshow("Live",frame)

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
if __name__ == '__main__':
    main()



#########################

#import cv2
#img=cv2.imread('images/25.jpg')
#cv2.imshow("25",img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,binary=cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
#cv2.imshow("binary",binary)
#element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
#fgmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)  # 开运算去噪
#contours,hierarchy=cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#ret,fgmask=cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
#cv2.drawContours(img,contours,-1,(0,0,255),3)

#cv2.connectedComponentsWithStats(fgmask)
#contours, hierarchy = cv2.findContours(binary, 2, 1)
#cnt = contours[0]
## 寻找凸包并绘制凸包（轮廓）
#hull = cv2.convexHull(cnt)

#length = len(hull)
#for i in range(len(hull)):
#    cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)

#cv2.imshow("img",img)
#cv2.waitKey(0)

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:24:57 2019

@author: xiepeng
"""
import cv2
import numpy as np
im=cv2.imread('images/25.jpg')
w,h,n= im.shape
im_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(im_gray,128,255,cv2.THRESH_BINARY)
cv2.imshow('th',thresh)
nccomps = cv2.connectedComponentsWithStats(thresh)#labels,stats,centroids
_ = nccomps[0]
labels = nccomps[1]
centroids = nccomps[3]
status = nccomps[2]
for row in range(status.shape[0]):
    if status[row,:][0] == 0 and status[row,:][1] == 0:
        background = row
    else:
       continue
status_no_background = np.delete(status,background,axis=0)
rec_value_max = np.asarray(status_no_background[:,4].max())
re_value_max_position = np.asarray(status_no_background[:,4].argmax())
h = np.asarray(labels,'uint8') 
h[h==(re_value_max_position+1)]=255
for single in range(centroids.shape[0]):
    print(tuple(map(int,centroids[single])))
    #position = tuple(map(int,centroids[single]))
    #cv2.circle(h, position, 1, (255,255,255), thickness=0,lineType=8)
cv2.imshow('h',h)
cv2.imshow('im_bw',thresh)
cv2.imshow('im_origin',im)
cv2.waitKey(0)
cv2.destroyAllWindows()


##############
import cv2
import time
import numpy as np

def line_detect_possible_demo(image):#霍夫变换找直线
        #通常用第二种方式。
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray1, 50, 150, apertureSize=3)
        #自动检测可能的直线，返回的是一条条线段
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
        print(type(lines))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("linepossible_demo", image)
def facesdetecter(image):
    start = time.time()#开始时间
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
    image=cv2.GaussianBlur(image,(5,5),0)
    image2=image.copy()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)#将图片转化成HSV格式
    #cv2.imshow("hsv",hsv)#显示HSV图
    H,S,V=cv2.split(hsv)
    #cv2.imshow("hsv-H",H)#显示HSV图明度
    #ret, thresh =cv2.threshold(H,17,25,0)#阈值分割
    thresh_h=cv2.inRange(H,30/2,50/2)#0-180du 提取人体肤色区域
    #ret,thresh_h=cv2.threshold(H, 30/2, 50/2, cv2.THRESH_BINARY)#二值化
    #ret,thresh_h=cv2.threshold(thresh_h, 155, 255, cv2.THRESH_BINARY)#二值化
    cv2.imshow("hsv-H-threshold",thresh_h)#显示二值化图
    #dilateh = cv2.dilate(thresh_h, None, iterations=2)#膨胀
    #cv2.imshow('dilateh', dilateh)
    #erode = cv2.erode(dilate, None, iterations=2)# 腐蚀
    #cv2.imshow('erodeh', erode)

    canny=cv2.Canny(gray,50,150)
    #cv2.imshow("canny",canny)#显示边缘处理图

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#人脸检测
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)#眼睛检测
    upperbody = upperbody_cascade.detectMultiScale(gray, 1.3, 5)#上身检测
    mouth=mouth_cascade.detectMultiScale(gray, 1.3, 5)#嘴巴检测
    nose=nose_cascade.detectMultiScale(gray, 1.3, 5)#鼻子检测
    #lefteye=lefteye_cascade.detectMultiScale(gray, 1.3, 5)#左眼检测
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in eyes:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #frame = cv2.rectangle(image,(x+10,y+10),(x+w,y+h),(0,255,0),2)
        #print(eyes)
        #print("--eyes------")
    for (x,y,w,h) in upperbody:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in mouth:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
    for (x,y,w,h) in nose:
        frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    #for (x,y,w,h) in lefteye:
        #frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,250),2)
    #计算帧率
    total_area_mask=0
    total_area_eyes=0
    if len(eyes)==2:
        #眼睛区域
        rect_eyes=[(eyes[0][0],eyes[0][1],eyes[1][0]+eyes[1][2]-eyes[0][0],eyes[1][1]+eyes[1][3]-eyes[0][1])]
        for (x,y,w,h) in rect_eyes:
            frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,250,255),2)
            thresh_eyes=thresh_h[y:y+h,x:x+w]
            contours, hierarchy = cv2.findContours(thresh_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #寻找前景 
            cv2.drawContours(image,contours,-1,(0,255,0),3)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算轮廓面积           
                total_area_eyes+=Area
        print("total_area_eyes=",total_area_eyes)
        #口罩区域
        rect_mask=[(x,y+h,w,h*2)]
        for (x,y,w,h) in rect_mask:
            frame = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
               
            thresh_mask=thresh_h[y:y+h,x:x+w]
            #image2[y:y+h,x:x+w]=thresh_h
            contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #寻找前景 
            cv2.drawContours(image,contours,-1,(0,0,255),3)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算轮廓面积           
                total_area_mask+=Area
        print("total_area_mask=",total_area_mask)
        #print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积
        if total_area_eyes<total_area_mask:
            print("------------无口罩")
        if total_area_eyes>total_area_mask:
            print("------------------戴口罩")

            #cv2.imshow("hsv-H-threshold-roi",thresh_h)#显示二值化图
    end = time.time()#结束时间
    fps=1 / (end-start)#帧率
    frame = cv2.putText(image,"fps:{:.3f}".format(fps),(3,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)#绘制
    cv2.imshow("face",image)#显示最终图片
    cv2.imshow("face_f",image2)#显示肤色检测图片
    #line_detect_possible_demo(image)#霍夫变换找直线

def mogseparate(image):
    fgmask = mog.apply(image)   
    ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("fgmask", fgmask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)    
    backgimage = mog.getBackgroundImage()
    #查找轮廓
    #contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    #cv2.drawContours(binary,contours,-1,(0,0,255),3)
    #binary = cv2.erode(binary, None, iterations=4)# 腐蚀
    cv2.imshow('erode', binary)
    cv2.imshow("backgimage", backgimage)
    cv2.imshow("frame", image)
    cv2.imshow("binary", binary)
    
def knnseperate(image):
    mog_sub_mask = mog2_sub.apply(image)
    knn_sub_mask = knn_sub.apply(image)
 
    cv2.imshow('original', image)
    cv2.imshow('MOG2', mog_sub_mask)
    cv2.imshow('KNN', knn_sub_mask)
def trackAvg(image,k_write):
    if k_write==1:
        cv2.imwrite("images/backgound.jpg",image)
        k_write=0 
    background=cv2.imread("images/backgound.jpg")
    #cv2.RunningAvg(image, background, 0.1, None)    
    #cv2.imshow('live', frame)
    #cv2.imshow('avg',avg_show)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar眼镜特征分类器'''

upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
upperbody_cascade.load("data/haarcascades/haarcascade_upperbody.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mouth_cascade.load("data/haarcascades/haarcascade_mcs_mouth.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
nose_cascade.load("data/haarcascades/haarcascade_mcs_nose.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

lefteye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
lefteye_cascade.load("data/haarcascades/haarcascade_lefteye_2splits.xml")#一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mog = cv2.createBackgroundSubtractorMOG2()
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

if __name__ == '__main__':
    

    k_write=1
    capture=cv2.VideoCapture(0) 
    while(True):
    
        ref,frame=capture.read()
        if ref==False:
            print("打开摄像头错误")
            break
        #cv2.imshow("frame",frame)
        #等待30ms显示图像，若过程中按“Esc”退出
        c= cv2.waitKey(30) & 0xff 
        if c==27:
            capture.release()
            break
        facesdetecter(frame)
        #mogseparate(frame)#mog方式分离前景
        #knnseperate(frame)#knn方式分离前景
        #trackAvg(frame,k_write)#runningavg分离前景
        
    cap.release()
    cv2.destroyAllWindows()