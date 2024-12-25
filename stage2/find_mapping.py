from datasets import load_dataset, Dataset
import re
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import json
def extract_pid(id_str):
    # Extract the numeric pid from the ID string
    match = re.search(r'(\d+)$', id_str)
    return int(match.group(1)) if match else -1

def get_category(id_str):
    if 'general' in id_str:
        return 0
    elif 'regional' in id_str:
        return 1
    elif 'suggestion' in id_str:
        return 2
    return 3  # for unexpected cases

def reorder_dataset(dataset):
    # Create list of (index, pid, category) tuples for sorting
    ordering = [(i, extract_pid(row['id']), get_category(row['id']))
               for i, row in enumerate(tqdm(dataset))]

    # Sort first by category, then by pid
    ordered_indices = [idx for idx, _, _ in sorted(ordering, key=lambda x: (x[2], x[1]))]

    # Select using the sorted indices
    return dataset.select(ordered_indices)


def compute_similarity(img1, img2) -> float:
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        return 0.0

    diff = np.abs(arr1 - arr2)
    # print(diff)
    
    mask = (diff.sum(axis=2) > 30).astype(np.uint8)
    # print (1 - (np.count_nonzero(mask) / mask.size))
    return 1 - (np.count_nonzero(mask) / mask.size)

def create_sequential_mapping(
    dataset: Dataset,
    id_range1: Tuple[int, int],  # Range for first sequence (1,2,3,...)
    id_range2: Tuple[int, int],  # Range for second sequence (1,1,1,2,...)
    similarity_threshold: float = 0.9,
    split: str = 'Val',
    image_cnt: int = 2084
) -> Dict[str, List[int]]:
    """
    Create mapping for sequential images where multiple ids in range2
    map to single ids in range1.
    Returns: Dict[id1 -> [id2_1, id2_2, ...]]
    """
    mapping = {}  # id1 -> list of matching id2s

    id2 = id_range2[0]

    duperr = 0
    maxerr_sim = 0
    for i in tqdm(range(id_range1[0], id_range1[1])):
        img1 = dataset[i]['image']

        matching_ids = []
        for j in range(id2, id_range2[1]):
            img2 = dataset[j]['image']
            similarity = compute_similarity(img1, img2)
            if similarity >= similarity_threshold:
                mapping[f'{split}_regional_{j-image_cnt}'] = f'{split}_general_{i}' 
                duperr = 0
                id2 += 1
            else:
                if maxerr_sim < similarity:
                    maxerr_sim = similarity
                duperr += 1
                if duperr >= 10:
                    print("Error: your condition is too harsh. Max err sim: ", maxerr_sim)
                    return mapping
                break



    return mapping

def save_mapping(mapping: Dict[str, List[int]], output_file: str):
    """Save mapping as JSON"""
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
split='Test'
image_cnt = 300
# Load and reorder
dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split.lower())
reordered_dataset = reorder_dataset(dataset)

mapping = create_sequential_mapping(
    reordered_dataset,
    id_range1=(0, image_cnt),    # First sequence
    id_range2=(image_cnt, len(reordered_dataset)-image_cnt),     # Second sequence
    split=split,
    image_cnt=image_cnt
)
save_mapping(mapping, "test_idmap.json")
