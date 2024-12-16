from ultralytics import YOLO
import gdown
import os

# Download and load the model
if not os.path.exists("yolo11x_1216.pt"):
    file_id = "1BSLZD9yk9YVjDpdPd2I7ExjFG2Cc06gn"  # Replace with your file ID
    output = "yolo11x_1216.pt"  # Your desired output filename
    gdown.download(id=file_id, output=output) 
    # gdown.download("https://drive.google.com/uc?id=1-_15b6X4fVQl-Hp4pQiY8s-YXjXjxqo", "best.pt", quiet=False)
model = YOLO("yolo11x_1216.pt")

# Load images
image_paths = ["/home/kcire/Music/yt/yolotest/datasets/coda/images/val/0003.jpg", "/home/kcire/Music/yt/yolotest/datasets/coda/images/val/0007.jpg"]

categories = {
    0: "person",
    1: "cyclist",
    2: "car",
    3: "truck",
    4: "tram",
    5: "tricycle",
    6: "bus",
    7: "bicycle",
    8: "moped",
    9: "motorcycle",
    10: "stroller",
    11: "wheelchair",
    12: "cart",
    13: "trailer",
    14: "construction_vehicle",
    15: "recreational_vehicle",
    16: "dog",
    17: "barrier",
    18: "bollard",
    19: "warning_sign",
    20: "sentry_box",
    21: "traffic_box",
    22: "traffic_cone",
    23: "traffic_island",
    24: "traffic_light",
    25: "traffic_sign",
    26: "debris",
    27: "suitcase",
    28: "dustbin",
    29: "concrete_block",
    30: "machinery",
    31: "chair",
    32: "phone_booth",
    33: "basket",
    34: "misc"
}


results = model(image_paths)
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Can change the below boxes.xyxy to boxes.xywh to xywh, xyxyn, xywhn
    # API reference : https://docs.ultralytics.com/modes/predict/#boxes

    print(f"image: {image_paths[i]}")
    for index, object in enumerate(boxes.cls):
        print(f"category_id: {object}, category: {categories[object.item()]},
               confidence: {boxes.conf[index].item()}, bbox: {boxes.xyxy[index]}")
    print("--------------------------------")
    
    
    
    result.show()  # display to screen
    result.save(filename=f"result_{i}.jpg")  # save to disk

