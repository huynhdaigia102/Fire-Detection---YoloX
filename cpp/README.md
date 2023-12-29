setup env
```bash
docker run --gpus all -itd --ipc=host --net=host --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/sda2/ExternalHardrive/research/object_detection/common/yolov5:/yolov5
  nvcr.io/nvidia/deepstream:5.0.1-20.09-devel
```

## Build

```
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```

## Export

```
./export ../pretrained/mnist.onnx ../pretrained/mnist.plan
```

## Infer

```
./infer pretrained/mnist.plan data/9.pgm
```