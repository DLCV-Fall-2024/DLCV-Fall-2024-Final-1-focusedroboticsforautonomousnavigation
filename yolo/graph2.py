import cv2
import torch
import numpy as np
import gdown
import os
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# Category mappings
categories = {
    0: "person", 1: "cyclist", 2: "car", 3: "truck", 4: "tram", 
    5: "tricycle", 6: "bus", 7: "bicycle", 8: "moped", 9: "motorcycle",
    10: "stroller", 11: "wheelchair", 12: "cart", 13: "trailer",
    14: "construction_vehicle", 15: "recreational_vehicle", 16: "dog",
    17: "barrier", 18: "bollard", 19: "warning_sign", 20: "sentry_box",
    21: "traffic_box", 22: "traffic_cone", 23: "traffic_island",
    24: "traffic_light", 25: "traffic_sign", 26: "debris", 27: "suitcase",
    28: "dustbin", 29: "concrete_block", 30: "machinery", 31: "chair",
    32: "phone_booth", 33: "basket", 34: "misc"
}

def create_gaussian_weight_mask(height, width, sigma=1.0):
    """Create a 2D Gaussian weight mask that emphasizes the center"""
    y = np.linspace(0, height-1, height)
    x = np.linspace(0, width-1, width)
    x, y = np.meshgrid(x, y)
    
    x0 = width // 2
    y0 = height // 2
    
    weights = np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))
    return weights / weights.sum()

def get_weighted_depth(depth_map, bbox):
    """Calculate weighted average depth with center emphasis"""
    x1, y1, x2, y2 = map(int, bbox)
    region_depth = depth_map[y1:y2, x1:x2]
    
    height, width = region_depth.shape
    weight_mask = create_gaussian_weight_mask(height, width, sigma=min(height, width)/4)
    
    weighted_depth = np.sum(region_depth * weight_mask)
    return weighted_depth

def generate_3d_spatial_graph(image_path, yolo_model, depth_pipe):
    # Load and process image
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    
    # Get depth map using transformer pipeline
    depth_map = depth_pipe(pil_image)["depth"]
    depth_map = np.array(depth_map)
    
    # Run object detection with custom model
    result = yolo_model(image)[0]
    
    # Create 3D regions
    height, width = image.shape[:2]
    x_bounds = [0, width/3, 2*width/3, width]
    y_bounds = [0, height/3, 2*height/3, height]
    
    # Get depth boundaries
    depth_values = depth_map.flatten()
    d_min, d_max = np.percentile(depth_values, [5, 95])
    d_bounds = [d_min, (d_min+d_max)/50, 2*(d_min+d_max)/3, d_max]
    
    def get_3d_region(bbox, depth):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # X position
        if center_x < x_bounds[1]:
            x_pos = "left"
        elif center_x < x_bounds[2]:
            x_pos = "middle"
        else:
            x_pos = "right"
            
        # Y position
        if center_y < y_bounds[1]:
            y_pos = "top"
        elif center_y < y_bounds[2]:
            y_pos = "middle"
        else:
            y_pos = "bottom"
            
        # Depth position
        if depth < d_bounds[1]:
            d_pos = "front"
        elif depth < d_bounds[2]:
            d_pos = "middle"
        else:
            d_pos = "back"
            
        return f"{d_pos} {x_pos} {y_pos}"
    
    # Map objects to 3D regions
    spatial_relations = {}
    
    boxes = result.boxes
    for idx, cls in enumerate(boxes.cls):
        bbox = boxes.xyxy[idx].cpu().numpy()
        class_id = int(cls.item())
        confidence = boxes.conf[idx].item()
        class_name = categories[class_id]
        
        weighted_depth = get_weighted_depth(depth_map, bbox)
        region = get_3d_region(bbox, weighted_depth)
        
        if region not in spatial_relations:
            spatial_relations[region] = []
        spatial_relations[region].append(f"{class_name} ({confidence:.2f})")
    
    # Format output
    formatted_output = []
    for region, objects in sorted(spatial_relations.items()):
        objects_str = ", ".join(objects)
        formatted_output.append(f"{region}: {objects_str}")
    
    return "\n".join(formatted_output)

def main():
    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Initialize depth estimation pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=DEVICE
    )
    
    # Download and load the YOLO model
    if not os.path.exists("yolo11x_1216.pt"):
        file_id = "1BSLZD9yk9YVjDpdPd2I7ExjFG2Cc06gn"
        output = "yolo11x_1216.pt"
        gdown.download(id=file_id, output=output)
    
    model = YOLO("yolo11x_1216.pt")
    
    # Process images
    image_paths = [
        "/home/kcire/Music/yt/yolotest/datasets/coda/images/val/0003.jpg",
        "/home/kcire/Music/yt/yolotest/datasets/coda/images/val/0007.jpg"
    ]
    
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        spatial_graph = generate_3d_spatial_graph(image_path, model, depth_pipe)
        print(spatial_graph)
        print("-" * 50)

if __name__ == "__main__":
    main()