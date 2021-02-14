import cv2
import numpy as np
import pickle
import sqlite3
import datetime 


faceDetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam= cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read('recognizer/trainingData.yml')
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
fontface = cv2.FONT_HERSHEY_SIMPLEX
font_small = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontscale = 1
fontcolor = (255, 255, 255)
color_small = (0, 255, 255)

stop = 1
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM Person WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


while(True):
    ret, img= cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if stop:
            time = datetime.datetime.now() 
            conn=sqlite3.connect('FaceBase.db')
            conn.execute("UPDATE Person SET time= ? WHERE ID = ?", (time, id))
            conn.commit()
            conn.close()
            stop=0
        
        if(profile!=None):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, profile[1], (x,y+h+30), fontface, fontscale, fontcolor) 
            cv2.putText(img, str(profile[2]), (x+430,y+h+30), fontface, fontscale, fontcolor) 
            cv2.putText(img, profile[3], (x+220,y+h+30), fontface, fontscale, fontcolor) 
            cv2.putText(img, profile[4], (x,y+h+70), font_small, fontscale, color_small)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img, 'Unknown', (x,y+h+30), fontface, fontscale, fontcolor) 
        #    cv2.PutText(img,profile[1],(x,y+h+30),font,255)
        #    cv2.PutText(img,profile[2],(x,y+h+60),font,255)
        #    cv2.PutText(img,profile[3],(x,y+h+90),font,255)
          
    cv2.imshow('face',img)
    k=cv2.waitKey(1)
    if(k==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
    
