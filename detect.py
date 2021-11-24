import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ret, t0 = cap.read()
ret, t1 = cap.read()

gray1 = cv2.cvtColor(t0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)

blur1 = cv2.GaussianBlur(gray1, (7,7), 0)
blur2 = cv2.GaussianBlur(gray2, (5,5), 0)

d = cv2.absdiff(blur1, blur2)

ret, thresh = cv2.threshold(d, 10, 255, cv2.THRESH_BINARY)

dilated = cv2.dilate(thresh, None, iterations=1)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

try:
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    markColor = (0,255,0)
    cv2.drawContours(t1, cnt, -1, markColor, 2)
    cv2.rectangle(t1, (x,y), (x+w, y+h), markColor, 2)
    cv2.imshow('t1', t1)
except ValueError:
    pass

if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()