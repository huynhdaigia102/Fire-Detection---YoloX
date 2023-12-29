'''
merge multiple data annotations with the following structure:
root
│   
└───annotations
│       anno1.json  // sub_dataset1.json + sub_dataset2.json 
│       ...
│
└───data_name
│   │
|   └───sub_dataset1
│   │       file012.jpg
│   │
│   └───sub_dataset1
│           file111.jpg
│           file112.jpg
'''

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


data_dir = Path("/mnt/nvme0n1/datasets/person_detect/anno_merges")
output_path = data_dir.joinpath("merge.json")
data = None
image_id = 1
anno_id = 1

for json_path in data_dir.glob("*.json"):
    data_name = json_path.stem
    print(f"INFO: json_path={json_path}...")
    with open(str(json_path), 'r') as f:
        json_data = json.load(f)
    if data is None:
        data = {
            "categories": json_data["categories"]
        }
    
    image_id_map = {}
    for img_info in json_data["images"]:
        image_id_map[img_info["id"]] = image_id
        img_info["id"] = image_id
        img_info["file_name"] = f"{data_name}/"+img_info["file_name"]
        image_id += 1
        
        images = data.get("images", [])
        images.append(img_info)
        data["images"] = images
    
    for anno in tqdm(json_data['annotations']):
        anno["image_id"] = image_id_map[anno["image_id"]]
        anno["id"] = anno_id
        anno_id+=1

        annotations = data.get("annotations", [])
        annotations.append(anno)
        data["annotations"] = annotations

print(len(data["images"]))
with open(str(output_path), 'w') as f:
    json.dump(data, f, indent=4)
print("Done...")