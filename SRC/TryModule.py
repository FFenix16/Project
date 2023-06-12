from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor



imgDir = "img/Testmodule/2.png" #Importing the img
model = YOLO('LOLAIP2.pt')  #Loading the model
results = model.predict(source=imgDir, save=True, imgsz=320, conf=0.3) #Run and save the model
