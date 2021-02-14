import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
val=0
imgsq=cv2.imread('assets/Admin2.jpg')
faceDetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True: 
    ret,img=cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        rect=img[y+h/2-100:y+h/2+100,x+w/2-100:x+w/2+100]
        cv2.addWeighted(rect,1,imgsq,1,0,rect)
        cv2.putText(img,"ADMIN",(x+h/2,y+w+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    
    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break
    
    cv2.imshow('Video', img)
