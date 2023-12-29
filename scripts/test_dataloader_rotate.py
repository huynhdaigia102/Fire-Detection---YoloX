import cv2
import numpy as np
import sys
import os
import math
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, "../"))

from yolox.exp import get_exp


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


if __name__ == "__main__":
    # args
    exp_file = "exps/fisheye/yolox_nano_rotate.py"
    name = None
    
    exp = get_exp(exp_file, name)
    exp.input_size = (1000, 1000)
    
    train_loader = exp.get_data_loader(
        batch_size=2,
        is_distributed=False,
        no_aug=False,
        cache_img=False,
    )
    
    datalen = len(train_loader.dataset)
    print(f"INFO: datalen={datalen}...")
    for i in range(datalen):
        index = np.random.randint(0, datalen)
        img, labels = train_loader.dataset[index][:2]
        img = np.transpose(img.astype(np.uint8), [1, 2, 0])
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        bbox = labels[:, 1:]
        bbox = bbox[bbox.sum(1)>0]
        # bbox[:, 4] = bbox[:, 4] / 180 * math.pi

        shape = img.shape[:2]
        vector_1 = np.array([[0, -10]]*bbox.shape[0])
        vector_2 = np.stack([bbox[:, 0]-shape[1]/2, bbox[:, 1]-shape[0]/2], axis=1)
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1, axis=1)[..., np.newaxis]
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2, axis=1)[..., np.newaxis]
        dot_product = np.array([np.dot(unit_vector_1[i], unit_vector_2[i]) for i in range(unit_vector_1.shape[0])])
        angle = np.arccos(dot_product)
        angle[vector_2[:,0]<0] *= -1
        bbox[:, 4] = angle
        
        bbox = npxywha2vertex(bbox)
        bbox = np.concatenate([np.zeros((bbox.shape[0], 1)), bbox], axis=1)
        
        for conf, *xyxyxyxy in bbox:
            xyxyxyxy = [int(i) for i in xyxyxyxy]
            pts = np.array([xyxyxyxy[i:i+2] for i in range(0, len(xyxyxyxy), 2)], np.int32)

            pts = pts.reshape((-1, 1, 2))
            cv2.putText(img, f'{conf:.2f}', (xyxyxyxy[0],xyxyxyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
            img = cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        
        # bboxs = labels[:, 1:]  # cxcywh
        # for cx, cy, w, h in bboxs:
        #     # if w==0 or h==0:
        #     #     continue
        #     # .astype(np.int32)
        #     cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0,0,255), 2)
        
        cv2.imshow("img", img)
        key = cv2.waitKey(0) & 0xff
        if key == ord("q"):
            break
    cv2.destroyAllWindows()