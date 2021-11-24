import cv2
import numpy as np

video = "D:/video.mp4"
cap = cv2.VideoCapture(video)

lower_white = np.array([78,25,221])
upper_white = np.array([125,99,255])

while(cap != 0):
    # 讀取影像到 frame 中，顏色識別方式轉換為 HSV
    ret, frame = cap.read()
    if ret == False:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg = cv2.blur(frame, (4,4))
    avg_float = np.float32(avg)
    
    # 模糊處理
    blur = cv2.blur(frame, (4,4))
    
    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(avg, blur)
    
    # 將圖片轉為灰階
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 篩選出變動程度大於門檻值的區域
    ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    # 使用型態轉換函數去除雜訊
    kernal = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal, iterations=2)
    
    # 產生等高線
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        # 忽略太小的區域
        if cv2.contourArea(c) < 2500:
            continue
        
        # TODO 偵測到物體 ...
        
        # 計算等高線的外框範圍
        (x, y, w, h) = cv2.boundingRect(c)
        
        # 劃出外框
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # 畫出等高線（除錯用）
    cv2.drawContours(frame, cnts, -1, (0,255,255), 2)
    
    cv2.namedWindow('fra', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('fra', frame)
    
    # 顯示原來的視訊
    cv2.namedWindow('frame', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)
    
    # 擷取視訊中白色的部分
    mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('mask', mask)
    
    # 原來視訊和擷取後的視訊操作
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.namedWindow('res', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('res', res)
    
    # q 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()