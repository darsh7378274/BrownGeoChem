print("script starting")

from ultralytics import YOLO

print("ultralytics imported successfully")

model = YOLO("yolov8n.pt")

print("model loaded")

results = model("test image.jpg")

print("inference done")

results[0].show()

print("done")