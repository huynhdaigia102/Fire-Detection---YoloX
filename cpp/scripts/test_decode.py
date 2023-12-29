import torch
import numpy as np
import timeit

from yolo._C import decode, nms


anchors = [
    [10,13], [16,30], [33,23],
    [30,61], [62,45], [59,119],
    [116,90], [156,198], [373,326]
]
anchors = np.array(anchors).reshape(3, -1)
strides = np.array([8, 16, 32])

data = np.random.uniform(size=(1, 18, 24, 40)).astype(np.float32)
data_tensor = torch.from_numpy(data.copy()).to("cuda")
score_thresh = 0.5
top_n = 250
score_thresh = 0.5
nms_thresh = 0.1
detections_per_im = 150
img_size = 640

for _ in range(100):
# while True:
    t0 = timeit.default_timer()
    scores, boxes = decode(data_tensor, anchors[0], top_n, score_thresh, strides[0])
    nms_scores, nms_boxes = nms(scores, boxes, nms_thresh, detections_per_im)
    
    h_scores = nms_scores.cpu().numpy()[0]
    h_boxes = nms_boxes.cpu().numpy()[0]
    t1 = timeit.default_timer()
    print(t1-t0)

print(h_scores, h_boxes)