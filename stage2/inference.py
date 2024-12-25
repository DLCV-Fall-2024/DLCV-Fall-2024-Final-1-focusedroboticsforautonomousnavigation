import gc
import torch
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from peft import PeftModel
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
import json
import time
import os
def clear_gpu_memory():
    gc.collect()  # Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Optional: even more aggressive
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
clear_gpu_memory()



### Load dataset
# dataset = load_dataset(
#     "ntudlcv/dlcv_2024_final1",
#     cache_dir="./cache",
#     download_mode="reuse_cache_if_exists",  # Correct value
#     num_proc=4
# )
# print(dataset)
save_dir = "/home/kcire/Music/rp/LLaVA/train_data"
# for data in dataset["train"]:
#     image = data["image"]
#     if os.path.isdir(save_dir) is False:
#         os.makedirs(save_dir)
#     if os.path.isdir(os.path.join(save_dir, "image")) is False:
#         os.makedirs(os.path.join(save_dir, "image"))
#     image.save(os.path.join(save_dir, "image", f"{data['id']}.jpg"))


### Load model
model_path = "liuhaotian/llava-v1.5-7b"
lora_path = "/home/kcire/Music/rp/LLaVA/stage1_lora_2/checkpoint-1400"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

model = PeftModel.from_pretrained(model, lora_path)


### Start inference

start_id = 3000
print("start_id:",start_id)
end_id = 4000
print("end_id:",end_id)

# Load existing responses if the file exists
responses_file = '/home/kcire/Music/rp/LLaVA/responses.json'
try:
    with open(responses_file, 'r') as f:
        responses = json.load(f)
        print("Loaded existing responses.")
        print("responses:",responses)
except (FileNotFoundError, json.JSONDecodeError):
    responses = {}

# Opening JSON file
with open('/home/kcire/Music/rp/LLaVA/processed_data_train.json') as f:
    datas = json.load(f)

print(datas[0])

for data in datas:
    # print(f"Current data ID: {data['id']}, Type: {type(data['id'])}")
    # print(f"Is ID in responses?: {data['id'] in responses}")
    # print(f"Current keys in responses: {list(responses.keys())[:5]}")  # Show first 5 keys
    
    if data["id"] < start_id or data["id"] > end_id:
        continue
    # Skip if already processed
    if str(data["id"]) in responses :
        print(f"Skipping {data['id']} - already processed")
        continue
    time1 = time.time()   
    image_path = os.path.join(save_dir, "image", f"Train_general_{data['id']}.jpg")
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config).to("cuda")
    prompt = data["conversations"][0]["value"][8:]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    time2 = time.time()
    print("converting prompt time:",time2 - time1)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.4,
            top_p=0.95,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=30,
            use_cache=True)
        time3 = time.time()
        print("generating time:",time3 - time2)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        time4 = time.time()
        print("decoding time:",time4 - time3)
    print("ID:",data["id"])
    print("Prompt:", prompt)
    print("Response:", outputs)
    
    # Save the new response
    responses[data["id"]] = outputs
    # with open(responses_file, 'a') as f:
    #     json.dump({data["id"]: outputs}, f)
    #     f.write('\n')
    # Save to file after each generation
    with open(responses_file, 'w') as f:
        json.dump(responses, f, indent=4)
