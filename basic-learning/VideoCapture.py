# 打开摄像头并灰度化显示
import cv2

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
print(capture.isOpened())
print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))    # 获取相机分辨率
ret = capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
ret = capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)# 重设相机分辨率为320*240
# capture.release()
while(True):
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # for i in range(frame.shape[0]):
    #     for j in range(frame.shape[1]):
    #         gray_inv = 255 - gray[i,j]
    #         gray[i,j] = gray_inv

    cv2.imshow("frame", gray)
    if cv2.waitKey(5) == ord('q'):
        capture.release()
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(25) == 27:
        capture.release()
        cv2.destroyAllWindows()
        break