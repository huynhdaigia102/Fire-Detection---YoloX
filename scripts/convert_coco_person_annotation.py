'''
convert coco anno to 1 cls
'''

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

json_path = Path("/mnt/sda1/datasets/coco/annotations/instances_val2017.json")
select_id = 1
output_path = json_path.parent.joinpath(json_path.stem + f"_{select_id}.json")
with open(str(json_path), 'r') as f:
    json_data = json.load(f)

annotations = []
anno_id = 0
categories = []
for anno in tqdm(json_data['annotations']):
    if anno['category_id'] == select_id:
        anno_id+=1
        anno["id"] = anno_id
        annotations.append(anno)
for cat in json_data['categories']:
    if cat['id'] == select_id:
        categories.append(cat)

json_data['annotations'] = annotations
json_data['categories'] = categories

with open(str(output_path), 'w') as f:
    json.dump(json_data, f, indent=4)
print("Done...")