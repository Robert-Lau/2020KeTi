import cv2
import MaskDetecterSystem as mask


def getbackground():
    cap = cv2.VideoCapture(0)
    num=0
    while True:
        ok, frame = cap.read()
        image=cv2.GaussianBlur(frame,(5,5),0)#高斯滤波
        #cv2.imshow("gauss",image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图片转化成灰度         
        num=num+1
        if num==10:
            backgound=gray
            cv2.imwrite("images/backgound.jpg",backgound)
            cv2.imshow("backgound",backgound)
            break
if __name__ == '__main__' :
    colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))#定义矩形颜
   
    cap = cv2.VideoCapture(0)
    getbackground()#保存背景
    backgound=cv2.imread("images/backgound.jpg")
    backgound=cv2.cvtColor(backgound,cv2.COLOR_BGR2GRAY)#将图片转化成灰度 
    gray=backgound#k-1张
    mask.facesdetecter_init()
    while True:
        gray_last=gray#k-1张
        ok, frame = cap.read()
        cv2.imshow("framelive",frame)
        if not ok:
            print('Cannot read video file')
            sys.exit()     
        image=cv2.GaussianBlur(frame,(5,5),0)#高斯滤波
        #cv2.imshow("gauss",image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图片转化成灰度 
        fgmask1=cv2.absdiff(gray,backgound)#背景差分
        cv2.imshow("fgmask1",fgmask1)
        fgmask2=cv2.absdiff(gray,gray_last)#帧间差分法
        cv2.imshow("fgmask2",fgmask2)
        fgmask=cv2.addWeighted(fgmask1,0.5,fgmask2,0.5,0)
        #fgmask=cv2.add(fgmask1,fgmask2)#差分均值
        #cv2.imshow("fgmask",fgmask)
        #fgmask=frame-backgound
        #cv2.imshow("chafen",fgmask)
        ret,fgmask=cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)#二值化
        #cv2.imshow("fgmask",fgmask)
        
       
        fgmask = cv2.dilate(fgmask, None, iterations=18)#膨胀
        #cv2.imshow('dilate', fgmask)
        fgmask = cv2.erode(fgmask, None, iterations=1)# 腐蚀
        #cv2.imshow('erode', fgmask)
        #frame = cv2.bitwise_and(frame, frame, mask=erode)#与操作

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))  # 形态学去噪
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, element)  # 开运算去噪
        #cv2.waitKey()
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #寻找前景 
        cv2.drawContours(frame,contours,-1,(0,0,255),3)     
        rect_array=[]
        count=0
        for cont in contours:
            Area = cv2.contourArea(cont)  # 计算轮廓面积
            if Area < 2000:  # 过滤面积小于10的形状
                continue

            count += 1  # 计数加一
                    
            print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积

        
            rect = cv2.boundingRect(cont) #提取矩形坐标
            rect_array.append(rect)
        
            #print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),colour[count%6],1)#原图上绘制矩形
            cv2.rectangle(fgmask1,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)  #黑白前景上绘制矩形

            y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
            cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号

    

        cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) #显示总数
        cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        #print("----------------------------")

    
        #cv2.imshow("bitwise_and",fgmask)
        cv2.imshow("findcont",frame)
        # quit on ESC button

        mask.facesdetecter(frame)
        


        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
