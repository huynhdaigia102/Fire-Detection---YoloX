import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


json_path = Path("/datasets/fisheye/merge/merge.json")
output_path = json_path.parent.joinpath(f"{json_path.stem}_normal.json")
print(f"INFO: json_path={json_path}...")
with open(str(json_path), 'r') as f:
    json_data = json.load(f)

# convert cx,cy,w,h,angle to xywh
for i, anno in enumerate(json_data['annotations']):
    bbox = json_data['annotations'][i]["bbox"][:4]
    w = bbox[2]
    h = bbox[3]
    if w > h:
        bbox[2] = h
        bbox[3] = w
    bbox[0] = bbox[0] - bbox[2] / 2
    bbox[1] = bbox[1] - bbox[3] / 2
    
    json_data['annotations'][i]["bbox"] = bbox

with open(str(output_path), 'w') as f:
    json.dump(json_data, f, indent=4)
print("Done...")