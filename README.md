# Introduction

Nhánh master được base từ tag 0.3.0: https://github.com/Megvii-BaseDetection/YOLOX

## Prepare dataset
```bash
cd <YOLOX_HOME>
ln -s /path/to/your/dataset_name ./datasets/dataset_name
```

## Train
```bash
python3 "tools/train.py" \
  -f "exps/releases/exp_name.py" \
  -b 64 -o --fp16 \
  --logger wandb wandb-project yolox-baseline
  # --resume -c "weights/exp_name/latest_ckpt.pth" --cache
```

## Test
```bash
python3 "tools/eval.py" \
  -f "exps/releases/exp_name.py" \
  --b 32 --conf 0.55 --nms 0.45 --fp16 --fuse --test  --tsize "288,480" \
  -c "weights/exp_name/best_ckpt.pth"
```

## Demo

```bash
python tools/demo.py video -f "[config file: exps/exp_name.py]" \
  -c "weights/exp_name/best_ckpt.pth" \
  --path path/to/your/video --conf 0.55 --nms 0.45 --tsize "288,480" --device gpu --save_result
```


## Export ONNX

yolox_nano.py: config file

```bash
python tools/export_onnx.py --output-name weights/best_ckpt.onnx \
  --img_size 288 480 \
  -f "exps/releases/exp_name.py" \
  -c "weights/best_ckpt.path" --no-onnxsim
```

## Export TensorRT
### X86
```bash
cd cpp && mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make -j4
./export [onnx_path] [trt_path]
./export_rotate [onnx_path] [trt_path]  // for rotate
```
### Jetson
```bash
cd cpp && mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" -DPLATFORM_TEGRA=ON ..
make -j4
./export [onnx_path] [trt_path]
./export_rotate [onnx_path] [trt_path]  // for rotate
```
### Benchmark
```bash
docker run --runtime nvidia -itd --name bmk-yolox -v /path/to/your/repo:/workspace nvcr.io/nvidia/pytorch:20.03-py3

/usr/src/tensorrt/bin/trtexec --loadEngine=/path/to/your/model.plan --plugins=/path/to/yolox/cpp/build/librapid_trt.so --fp16 --explicitBatch
```
###### FPS = (1 / Mean GPU Compute) x 1000

## Model zoo


S3 storage: https://drive.google.com/drive/folders/17bx88f3txRUA2zTNb2CRRB7t5LG5SDQR

### Fire Detection
| Backbone                        |    Train / Eval Dataset    |             Exps         |         Input Shape                     | mAP</br>@[IoU=0.50:0.95] | GTX1650  | Jetson Nano |
| ------------------------------- | :--------------: | :----------------------------: | :--------------------------:  | :--------------------------: | :-------------------: | :----------------------------: |
| s_coco.pth           |      (1)       |         s_coco.py         |                |           |                     |                              |
| tiny_coco.pth           |      (1)       |         tiny_coco.py         |                        |      |                     |                              |
| nano_coco.pth           |      (1)       |         nano_coco.py         |                      |        |                     |                              |
| nano_220407.pth           |      (1)       |         nano_220316.py         |                       |       |                     |                              |
| nano_20220627.pth               |      (1)       |         nano_220316.py         |                  |            |                     |                              |
| nano_20221102.pth               |      (2)       |         nano_220316.py         |    480x288        | 0.46 |          2.5ms ms;</br>400 FPS          |               22.83 ms;</br>43.8 FPS  

### CXView
| Backbone                        |    Train / Eval Dataset    |             Exps         |         Input Shape                     | mAP</br>@[IoU=0.50:0.95] | GTX1650  | Jetson Nano |
| ------------------------------- | :--------------: | :----------------------------: | :--------------------------:  | :--------------------------: | :-------------------: | :----------------------------: |
| nano_220316.pth           |      (1)       |         nano_220316.py         |                |           |                     |                              |
| nano_220330.pth           |      (1)       |         nano_220316.py         |                        |      |                     |                              |
| nano_220407.pth           |      (1)       |         nano_220316.py         |                      |        |                     |                              |
| nano_220407.pth           |      (1)       |         nano_220316.py         |                       |       |                     |                              |
| nano_20220627.pth               |      (1)       |         nano_220316.py         |                  |            |                     |                              |
| nano_20221102.pth               |      (2)       |         nano_220316.py         |    480x288        | 0.46 |          2.5ms ms;</br>400 FPS          |               22.83 ms;</br>43.8 FPS               |


(1): CrowndHuman, SCID, GenViet, KnK, Canifa, TSN

(2): CrowndHuman, SCID, GenViet, KnK, Canifa, TSN + Thiso Mall



## Fisheye (Pharmacity)

| Backbone                        |    Train / Eval Dataset    |             Exps        | Input Shape     | mAP</br>@[IoU=0.50:0.95] | GTX1650  | Jetson Nano |
| ------------------------------- | :--------------: | :----------------------------: | :--------------------------: | :--------------------------: | :-------------------: | :----------------------------: |
| nano_rotate_20220818.pth | Pharmacity | nano_fisheye_220818.py |           480x480           | Val: 0.43 |                | 56.4 ms; 17.7 FPS |
| nano_rotate_20230704.pth | Pharmacity | nano_fisheye_220818.py |           480x480           | Val: 0.43 |                | 56.4 ms; 17.7 FPS |


## Window Shop (Pharmacity)

| Backbone                        |    Train / Eval Dataset    |             Exps       | Input Shape       | mAP</br>@[IoU=0.50:0.95] | GTX1650  | Jetson Nano |
| ------------------------------- | :--------------: | :----------------------------: | :--------------------------: | :--------------------------: | :-------------------: | :----------------------------: |
| nano_20230323.pth | (1) | nano_230323.py |             480x288             |        |     2.5ms ms;</br>400 FPS     |22.83 ms;</br>43.8 FPS|
| nano_wds_20230620.pth | (2) | nano_wds_20230620.py |        640x384        | Val (20230620_WDS_PMC): 0.44 </br> Test (20220315_WDS_PMC): 0.57 |     3.15ms; 301 FPS     |37.4 ms; 26.73 FPS|


(1): CrowndHuman, SCID, GenViet, KnK, Canifa, TSN + Pharmacity

(2): Coco + Pharmacity


## Changelogs

**2023/06/20**: Update model
  * Train new model 20230620_WDS_PMC: Dữ liệu khu vực cửa sổ

**2023/03/23**: Update model
  * Bổ sung 20220315_WDS_PMC: Dữ liệu khu vực cửa sổ

**2022/11/02**: Update model
  * Bổ sung genviet: aeonHD, bigcVing, oceanPark

**2022/06/07**: Update rotate
  * Tính cost dùng iou rotate
  * loss giữ nguyên iou thẳng
  * Sử dụng eval thẳng để đánh giá

**2022/06/23**: Update yolox parser plugin
  * Đưa property muxer-width, muxer-height. Default = (1920, 1080)
  * Hỗ trợ fisheye (phát hiện nếu có góc)

**2022/06/27**: Update release exp: nano_220316 (migrate latest version)

**2022/07/14**: 
  * Fix bug export model fisheye code.
  * Update fisheye metadata: detail in [README](deepstream/plugins/README.md)

**2022/08/19**: 
  * Refactor code. yolox:0.3.0
  * Update fisheye metadata name: FisheyeData
  * Release person_yolox_nano_rotate: person_nano_rotate_20220818.pth
