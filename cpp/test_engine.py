import os
import sys
import torch
import cv2
import numpy as np
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, ".."))

from yolo._C import Engine


def preprocess(image, target_size):
    h, w = image.shape[:2]
    d = max(h, w)
    resize = float(target_size / d)
    img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    dx, dy = int(target_size - w), int(target_size - h)
    img = cv2.copyMakeBorder(img, 0, dy, 0, dx, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = img.transpose(2, 0, 1)
    img = np.float32(img)

    # import torch
    # pad_info = torch.Tensor((w, h) + (0, 0) + (w*resize, h*resize))
    
    return img, resize


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # coords[:, [0, 2]] -= pad[0]  # x padding
    # coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    return coords


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, 0, bottom+top, 0, right+left, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class YoloDetector:
    def __init__(self, weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Engine.load(weights)
        print("Load success..")
        self.stride = 32 # model stride
        self.img_size = 640  # multiple of the strides
    
    def detect(self, image):
        # Padded resize
        img_resize = letterbox(image, self.img_size, stride=self.stride)[0]
        # Convert
        img = img_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        print(img.shape)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        scores, boxes = self.model(img)
        scores = scores.clone().cpu().numpy()[0]
        boxes = boxes.clone().cpu().numpy()[0]
        
        # det = pred[0]
        boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], image.shape).round()

        # for i in range(boxes.shape[0]):
        #     if scores[i] < 0.35: continue
        #     color = (0, 0, 255)
        #     x1, y1, x2, y2 = boxes[i][:4].astype(np.int32)
        #     cv2.rectangle(img_resize, (x1, y1), (x2, y2), color, 2)
        
        # # if bboxs.shape[0]: cv2.imwrite("frame.jpg", frame)
        # cv2.imshow("img_resize", img_resize)
        # key = cv2.waitKey(1) & 0xff
        # if key == ord("q"):
        #     break

        # print(scores, scores.shape, boxes.shape)
        return scores, boxes

if __name__ == '__main__':
    from imutils.video import VideoStream, FileVideoStream
    import imutils
    
    vs = VideoStream("rtsp://localhost:8554/stream1").start()
    yd = YoloDetector("/research/object_detection/common/yolov5/weights/best.plan")

    while True:
        frame = vs.read()
        # frame = imutils.rotate_bound(frame, 90)
        if frame is None:
            break
        
        frame = frame.copy()
        scores, boxes = yd.detect(frame)
        for i in range(boxes.shape[0]):
            if scores[i] < 0.45: continue
            color = (0, 0, 255)
            x1, y1, x2, y2 = boxes[i][:4].astype(np.int32)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # if bboxs.shape[0]: cv2.imwrite("frame.jpg", frame)
        cv2.imshow("frame", imutils.resize(frame, height=1000))
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    vs.stop()