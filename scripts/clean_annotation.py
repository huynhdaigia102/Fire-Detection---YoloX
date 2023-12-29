import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import shutil


data_dir = "datasets/20230620_WDS_PMC"

for json_path in glob(os.path.join(data_dir, "annotations","*.json")):

    data = None
    image_id = 1
    anno_id = 1

    print(f"INFO: json_path={json_path}...")
    phase_name = json_path.split("/")[-1].split(".")[0]
    output_path = os.path.join(data_dir, f"annotations/{phase_name}_clean.json")
    clean_dir = os.path.join(data_dir, f"{phase_name}_clean")
    os.makedirs(clean_dir, exist_ok=True)

    with open(str(json_path), 'r') as f:
        json_data = json.load(f)
    if data is None:
        data = {
            "categories": json_data["categories"]
        }

    image_id_map = {}
    for img_info in tqdm(json_data["images"]):
        image_id_map[img_info["id"]] = image_id
        img_info["id"] = image_id
        img_info["file_name"] = img_info["file_name"]
        image_id += 1
        
        images = data.get("images", [])
        images.append(img_info)
        data["images"] = images

        img_src = os.path.join(data_dir, img_info["file_name"])
        img_dst = os.path.join(clean_dir, img_info["file_name"].split("/")[-1])
        shutil.copy(img_src, img_dst)
    
    for anno in tqdm(json_data['annotations']):
        if any([val < 0 for val in anno["bbox"]]):
            continue
            print(anno)

        anno["image_id"] = image_id_map[anno["image_id"]]
        anno["id"] = anno_id
        anno_id+=1

        annotations = data.get("annotations", [])
        annotations.append(anno)
        
        data["annotations"] = annotations

    print(f'dataset {phase_name} = {len(data["images"])}')
    with open(str(output_path), 'w') as f:
        json.dump(data, f, indent=4)