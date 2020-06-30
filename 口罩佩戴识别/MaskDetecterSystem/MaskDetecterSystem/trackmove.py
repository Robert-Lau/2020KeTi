#import sys,os
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
#sys.path.append(BASE_DIR)
import numpy as np
import cv2
import time
import datetime
import tracktarget as tar

def trackmove(frame):
    frame=cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    fgmask = fgbg.apply(gray)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪
    ret,fgmask=cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)#二值化

    
    dilate = cv2.dilate(fgmask, None, iterations=4)#膨胀
    cv2.imshow('dilate', dilate)
    erode = cv2.erode(dilate, None, iterations=2)# 腐蚀
    cv2.imshow('erode', erode)

    contours, hierarchy = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #寻找前景 
    cv2.drawContours(frame,contours,-1,(0,0,255),3) 
    #cv2.drawContours(erode,contours,-1,(0,0,255),3) 
    cv2.imshow('erode2', erode)
    count=0
    rect_array=[]
    for cont in contours:
        Area = cv2.contourArea(cont)  # 计算轮廓面积
        if Area < 500:  # 过滤面积小于10的形状
            continue

        count += 1  # 计数加一

        print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积

        rect = cv2.boundingRect(cont) #提取矩形坐标
        rect_array.append(rect)
        
        print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),colour[count%6],1)#原图上绘制矩形
        cv2.rectangle(erode,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)  #黑白前景上绘制矩形

        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
        cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号

    

    cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) #显示总数
    cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    print("----------------------------")

    cv2.imshow('frame', frame)#在原图上标注
    #cv2.imshow('frame2', erode)  # 以黑白的形式显示前景和背景
    #out.write(frame)
    return rect_array

colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))#定义矩形颜色

fgbg = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法

#fourcc = cv2.VideoWriter_fourcc(*'XVID')#设置保存图片格式
#out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (768,576))#分辨率要和原视频对应

#frame=cv2.imread("images/26.jpg")

if __name__ == '__main__' :
    #cap = cv2.VideoCapture(0) #参数为0是打开摄像头，文件名是打开视频
    #cap = cv2.VideoCapture("images/vtest.avi") #参数为0是打开摄像头，文件名是打开视频
    cap = cv2.VideoCapture("images/face2.mp4")
    while True:
        ret, frame = cap.read()  #读取图片
    
        cv2.imshow("live",frame)

        trackmove(frame)

        k = cv2.waitKey(30)&0xff  #按esc退出
        if k == 27:
            break
    

    #out.release()#释放文件
    cap.release()
    cv2.waitKey()
    cv2.destoryAllWindows()#关闭所有窗口