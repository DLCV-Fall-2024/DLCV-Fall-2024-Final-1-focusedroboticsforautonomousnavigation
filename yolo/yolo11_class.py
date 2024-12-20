from ultralytics import YOLO
import gdown
import os

model = YOLO("yolo11x.pt")

class_names = model.names

# Print all classes
for id, name in class_names.items():
    print(f"{id}: {name}")
    with open("yolo11_class.txt", "a") as f:
        f.write(f"{id}: {name}\n")

