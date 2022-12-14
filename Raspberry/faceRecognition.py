import cv2
import numpy as np
import os
from datetime import datetime
from datetime import date
import requests
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


#path ke api

api_key_value = "randomapi190"

url = "http://192.168.43.229/iot-absenface//absen/absen"



id = 0

# NIP pegawai, kosongkan id 0 dan isi pake "none:

iddata = ['None', '1804030129', '1231243', '098120938', '1987123', '786593'] 

cam = cv2.VideoCapture(0)
cam.set(3, 640) # lebar video
cam.set(4, 480) # tinggi video

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100) and (confidence <40):
            id = iddata[id]
            confidence = "  {0}%".format(round(100 - confidence))
            data = "api_key=" + api_key_value + "&nip=" + id  +  " "
            #notifikasi suara wajah dikenali
            os.system('mpg321 absenberhasil.mp3 &')
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582",
                "Content-Type": "application/x-www-form-urlencoded"}
            requests.post(url,data,headers=headers)
            response = requests.post(url,headers,data)
            print("hallo", id)
            time.sleep(5)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
