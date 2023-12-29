import cv2
import numpy as np
import sys
import os
pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(pwd, "../"))

from yolox.exp import get_exp


if __name__ == "__main__":
    # args
    exp_file = "exps/releases/nano_wds_230706.py"
    name = None
    
    exp = get_exp(exp_file, name)
    
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
        
        bboxs = labels[:, 1:]  # cxcywh
        for cx, cy, w, h in bboxs:
            # if w==0 or h==0:
            #     continue
            # .astype(np.int32)
            cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0,0,255), 2)
        
        cv2.imshow("img", img)
        key = cv2.waitKey(0) & 0xff
        if key == ord("q"):
            break
    cv2.destroyAllWindows()