# 拖动鼠标绘制曲线或矩形
import cv2 as cv
import  numpy as np

drawing = False    # 按下鼠标，为真
mode = True        # 按m切换模式为绘制圆
ix,iy = -1,-1

# 鼠标回调函数
def mouse_draw(event,x,y,flags,param):
    global drawing,ix,iy,mode
    if event == cv.EVENT_LBUTTONDOWN: # 按下鼠标左键
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE: # 移动鼠标
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(255,0,0),-1)
            else:
                cv.circle(img,(x,y),5,(255,0,0),-1)

    elif event == cv.EVENT_LBUTTONUP:  # 移动完毕
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
        else:
            cv.circle(img, (x, y), 5,(255, 0, 0), -1)

img = np.zeros((512,512,3),np.uint8)
cv.namedWindow("MouseDraw")
cv.setMouseCallback("MouseDraw",mouse_draw)
while True:
    cv.imshow("MouseDraw",img)
    if cv.waitKey(20) & 0xff == 27:
        break
    elif cv.waitKey(20) == ord('m'):
        mode = not mode
        if mode == True:
            print("Mode:rectangle")
        elif mode == False:
            print("Mode:curve")

cv.destroyAllWindows()
