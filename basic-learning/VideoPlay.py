# 视频播放
# 与从相机捕获相同，只是用视频文件名更改摄像机索引。
import cv2 as cv
cap = cv.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\robert.avi")
print(cap.isOpened())

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("无法接收帧，正在退出...")
        break
    cv.imshow("robert",frame)
    if cv.waitKey(1) == ord("q"):# waitkey的参数如果很小，为加速视频；
                                 # 过大，为慢速视频
                                 #25ms，为正常速度
        break

cap.release()
cv.destroyAllWindows()