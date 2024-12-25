import json

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

def merge_json_files(file1_path, file2_path, output_path):
    # Read first file
    with open(file1_path, 'r') as f1:
        data1 = json.load(f1)
    
    # Read second file
    with open(file2_path, 'r') as f2:
        data2 = json.load(f2)
    
    # Merge data (data2 will override data1 for duplicate keys)
    merged_data = {**data1, **data2}
    
    # Write merged data to output file
    with open(output_path, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)


# with open("../stage2_train_with_general/processed_data.json", "r") as f:
#     datas = json.load(f)

# print(datas[-1]['conversations'][0]['value'])

# print(extract_example_response(datas[-1]['conversations'][0]['value']))

merge_json_files('../data/responses_0-4883.json', '../data/response_4884-6883.json', '../data/responses_all.json')