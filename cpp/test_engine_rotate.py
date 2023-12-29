import os
import sys
import torch
import cv2
import numpy as np
import timeit
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, ".."))

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
from bytetrack._C import EngineRotate, decode_rotate, nms_rotated


def npxywha2vertex(box):
    """
    use radian
    X=x*cos(a)-y*sin(a)
    Y=x*sin(a)+y*cos(a)
    """
    batch = box.shape[0]

    center = box[:,:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = np.empty((batch,2), dtype=np.float32)
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)

    hori = np.empty((batch,2), dtype=np.float32)
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)

    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)


class YoloDetector:
    def __init__(self, weights):
        exp_file = os.path.join(pwd, "../exps/default/yolox_nano_rotate.py")
        self.exp = get_exp(exp_file, exp_name=None)
        self.num_classes = 1
        self.preproc = ValTransform(legacy=False)
        self.exp.test_size = (480, 480)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EngineRotate.load(weights)
        print("Load success..")

    
    def detect(self, image):
        img_size = self.exp.test_size
        img_h, img_w = image.shape[:2]
        img, ratio = self.preproc(image, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        scores, boxes = self.model(img)
        nms_scores = scores.clone().cpu().numpy()
        nms_boxes = boxes.clone().cpu().numpy()

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        nms_boxes[..., :4] /= scale

        return nms_scores[0], nms_boxes[0]


if __name__ == '__main__':
    from imutils.video import VideoStream, FileVideoStream
    import imutils
    
    # vs = VideoStream("rtsp://localhost:8554/stream1").start()
    vs = FileVideoStream("/datasets/fisheye/20220411/videos/Sgpmc065-10.mp4").start()
    yd = YoloDetector("/research/object_detection/common/yolox/weights/yolox_nano_rotate/20220620.plan")

    while True:
        frame = vs.read()
        # frame = imutils.rotate_bound(frame, 90)
        if frame is None:
            break
        
        frame = frame.copy()
        scores, boxes = yd.detect(frame)

        boxes = npxywha2vertex(boxes)

        for i in range(boxes.shape[0]):
            if scores[i] < 0.4: continue

            xyxyxyxy = [int(i) for i in boxes[i]]

            pts = np.array([xyxyxyxy[i:i+2] for i in range(0, len(xyxyxyxy), 2)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        
        # if bboxs.shape[0]: cv2.imwrite("frame.jpg", frame)
        cv2.imshow("frame", imutils.resize(frame, height=1000))
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()