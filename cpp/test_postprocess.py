import torch
import numpy as np
import timeit
import sys
import os
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, ".."))

from rapid._C import Engine, decode, nms_rotated
from models.rapid import PostProcess

anchors = [
    [18.7807, 33.4659], [28.8912, 61.7536], [48.6849, 68.3897],
    [45.0668, 101.4673], [63.0952, 113.5382], [81.3909, 134.4554],
    [91.7364, 144.9949], [137.5189, 178.4791], [194.4429, 250.7985]
]
anchors = np.array(anchors)
data = np.random.uniform(size=(1, 18, 38, 38)).astype(np.float32)

top_n = 250
score_thresh = 0.5
nms_thresh = 0.1
detections_per_im = 150
img_size = 608
data_tensor = torch.from_numpy(data.copy()).to("cuda")
ex_anchors = anchors.copy().reshape(3,-1)[2]
for _ in range(10):
# while True:
    t0 = timeit.default_timer()
    scores, boxes = decode(data_tensor, ex_anchors, top_n, score_thresh, img_size)
    nms_scores, nms_boxes = nms_rotated(scores, boxes, nms_thresh, detections_per_im)
    
    h_scores = scores.cpu().numpy()[0]
    h_boxes = boxes.cpu().numpy()[0]
    t1 = timeit.default_timer()
    print(t1-t0)

# 
# pp = PostProcess(anchors[[6,7,8]])
# dts = pp.process(data)[0]
# order = dts[:,5].argsort()[::-1][:top_n]
# dts = dts[order]
# py_boxes = dts[:,:5]
# py_scores = dts[:,5]

# print(dts.shape)
