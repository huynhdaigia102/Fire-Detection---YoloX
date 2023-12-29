import os
import sys
import torch
import cv2
import numpy as np
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, ".."))

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


class YoloXDetector:
    def __init__(self, weights):
        # self.stride = int(self.model.stride.max())  # model stride
        exp_file = os.path.join(pwd, "../exps/person/yolox_nano.py")
        self.exp = get_exp(exp_file, exp_name=None)
        self.num_classes = 1
        self.preproc = ValTransform(legacy=False)
        self.exp.test_conf = 0.8
        self.exp.test_size = (576, 960)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.exp.get_model().to(self.device)
        ckpt = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    
    def detect(self, image):
        img_size = self.exp.test_size
        img_h, img_w = image.shape[:2]
        img, ratio = self.preproc(image, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.exp.test_conf, self.exp.nmsthre
            )
        outputs = outputs[0].cpu().numpy()
        scores = outputs[:, 4] * outputs[:, 5]
        bboxes = outputs[:, :4]  # x1y1x2y2

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # (n_objs, 6), *xyxy, conf, cls format
        return bboxes, scores


if __name__ == '__main__':
    from imutils.video import VideoStream, FileVideoStream
    import imutils
    
    # vs = VideoStream("rtsp://localhost:8554/stream1").start()
    vs = FileVideoStream("/mnt/nvme0n1/datasets/person_detect/scid/scid-20220610/videos/Cam3_20220608_17h46m.ts").start()
    yd = YoloXDetector(os.path.join(pwd, "../weights/person/nano_20220627.pth"))
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
            # if score < 0.6:
            #     continue
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