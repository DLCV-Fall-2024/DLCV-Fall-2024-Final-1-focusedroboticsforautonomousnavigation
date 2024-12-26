import gc
from weakref import ref
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import json
import os
from datasets import load_dataset
# from ultralytics import YOLO
from stage2.retrieval_YOLOCLIP_new import ImageRetrieval



# def setup_yolo(model_path="/home/jasper0314/YYS/DLCV-Fall-2024-Final-1-focusedroboticsforautonomousnavigation/YYS/yolo11x_1216.pt"):
#     """Initialize YOLO model"""
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"YOLO model not found at {model_path}")
#     return YOLO(model_path, task='detect')

# def YOLO_inference(image, model):
#     """Run YOLO inference on an image"""
#     with torch.no_grad():
#         results = model(image, stream=False)
#         return [[result.names[result.boxes.cls[i].item()]
#                 for i in range(result.boxes.cls.shape[0])] for result in results]

def extract_example_response(text):
    start_marker = "Example Response from a Similar Scene:"
    end_marker = "Question:"
    
    try:
        start_idx = text.index(start_marker) + len(start_marker)
        end_idx = text.index(end_marker)
        
        # Extract and trim whitespace/newlines
        example_response = text[start_idx:end_idx].strip()
        return example_response
    except ValueError:
        return None

def process_dataset(dataset, yolo_model, retriever: ImageRetrieval, output_dir, test=True):
    """Process dataset with YOLO model and image retrieval"""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    processed_data = []
    # with open("data/processed_data.json", "r") as f:
    #     ref_captions = json.load(f)
    with open("stage2/test_idmap.json", "r") as f:
        idmap = json.load(f)
    with open("stage1_test_general/responses.json", "r") as f:
        responses = json.load(f)

    for idx, item in enumerate(tqdm(dataset)):
        # if idx > 5:
        #     break
        # elif idx >= 20:
        #     break
        # Get image and save it
        image = item["image"]
        image_filename = os.path.join("images", f"image_{idx}.jpg")
        
        # Save original image
        # image.save(os.path.join(output_dir, "images", image_filename))
        
        # Process with YOLO
        # yolo_results = YOLO_inference(image, yolo_model)
        
        # Get similar images using retriever
        similar_ids, ref_caption = retriever.retrieve(item["id"], image, k=2, category="general")
        # original_prompt = ref_captions[idx]['conversations'][0]['value']
        # ref_caption = extract_example_response(original_prompt)
        # print(similar_ids)
        if "regional" in item["id"]:
            image_id = idmap[item["id"]].split("_")[-1]
        else:
            image_id = item["id"].split("_")[-1]
        
        if image_id not in responses:
            global_description = "No global description available."
        else:
            global_description = responses[image_id]
        
        # Determine question type
        if "general" in item["id"]:
            question_type = 0
        elif "regional" in item["id"]:
            question_type = 1
        else:
            question_type = 2
        
        # Create system and question prompts
        system_prompt = ("There is an image of traffic captured from the perspective of the ego car. "
                        "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), "
                        "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, "
                        "directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, "
                        "miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the "
                        "seven categories above.")
        
        # question_prompt = "Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
        question_prompts = ["Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.",
                       "Please describe the object inside the red rectangles (bounding boxes) in the image and explain why it affect ego car driving.",
                       "Please provide driving suggestions for the ego car based on the current scene."]

        # Create data entry
        if test:
            data_entry = {
                "id": idx,
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": (f"<image>\n{system_prompt}\n\n"
                                f"Global Description: \n\"{global_description}\"\n\n"  # Insert this in my prompt
                                f"Example Response from a Similar Scene:\n\"{ref_caption}\"\n\n"
                                f"Question: {question_prompts[question_type]}\n\n"
                                f"Answer:")
                    }
                ]
            }
        else:
            data_entry = {
                "id": idx,
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": (f"<image>\n{system_prompt}\n\n"
                                f"Global Description: \n\"{global_description}\"\n\n"  # Insert this in my prompt
                                f"Example Response from a Similar Scene:\n\"{ref_caption}\"\n\n"
                                f"Question: {question_prompts[question_type]}\n\n"
                                f"Answer:")
                    },
                    {
                        "from": "gpt",
                        "value": item["conversations"][1]["value"]
                    }
                ]
            }
        processed_data.append(data_entry)



    # Save processed data to JSON
    with open(os.path.join(output_dir, "processed_data.json"), 'w') as f:
        json.dump(processed_data, f, indent=4)

def main():
    # Load dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", num_proc=4)
    
    # Initialize YOLO model
    # yolo_model = setup_yolo()
    
    # Initialize image retriever
    retriever = ImageRetrieval(
        max_objects=5,
        yolo_model="checkpoint/yolo11x.pt",
        golden_json="stage2/golden.json",
        mapping_json="stage2/train_idmap.json"
    )
    
    # Process training data
    output_dir = "./stage2_test_general"
    process_dataset(dataset["test"], None, retriever, output_dir, test=True)
    
    # Process validation data
    # output_dir = "./stage2_val_general"
    # process_dataset(dataset["val"], yolo_model, retriever, output_dir)

if __name__ == "__main__":
    main()
