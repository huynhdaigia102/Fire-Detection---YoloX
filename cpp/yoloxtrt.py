import os
import sys
import torch
import cv2
import numpy as np
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, ".."))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from bytetrack._C import Engine


class YoloXDetectorTrt:
    def __init__(self, weights):
        # self.stride = int(self.model.stride.max())  # model stride
        self.test_size = (288, 480)  # h,w must multiple of the strides
        exp_file = os.path.join(pwd, "../exps/example/mot/yolox_nano_mix_det.py")
        self.exp = get_exp(exp_file, exp_name=None)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Engine.load(weights)

    
    def detect(self, image):
        img_size = self.exp.test_size
        img_h, img_w = image.shape[:2]
        img, ratio = preproc(image, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        scores, bboxes = self.model(img)
        scores = scores.clone().cpu().numpy()[0]
        bboxes = bboxes.clone().cpu().numpy()[0]

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # (n_objs, 6), *xyxy, conf, cls format
        return bboxes, scores


if __name__ == '__main__':
    from imutils.video import VideoStream, FileVideoStream
    import imutils
    
    vs = VideoStream("rtsp://localhost:8554/stream1").start()
    yd = YoloXDetectorTrt(os.path.join(pwd, "../weights/bytetrack_nano.plan"))
    # self.exp.test_conf
    while True:
        frame = vs.read()
        # frame = imutils.rotate_bound(frame, 90)
        if frame is None:
            break
        
        frame = frame.copy()
        bboxes, scores = yd.detect(frame)
        for i in range(bboxes.shape[0]):
            score = scores[i]
            if score < 0.6:
                continue
            color = (0, 0, 255)
            x1, y1, x2, y2 = bboxes[i][:4].astype(np.int32)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # if bboxs.shape[0]: cv2.imwrite("frame.jpg", frame)
        cv2.imshow("frame", imutils.resize(frame, width=1000))
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()