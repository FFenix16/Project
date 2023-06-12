from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


#def Learing():
imgDir = "img/Testmodule/1.png"
model = YOLO('LOLAIP2.pt')
results = model.predict(source=imgDir, save=True, imgsz=320, conf=0.3)

    
#def SetImg(imageDir):
 #   imageDir = '/runs/detect/predict/screenshot.png'
  #  return imageDir
    
