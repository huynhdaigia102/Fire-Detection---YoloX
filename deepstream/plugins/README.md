# Build

```bash
mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-5.0 \
    -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" \
    -DPLATFORM_TEGRA=ON ..
make && make install
```

# Run

```bash
URI=file:///research/object_detection/common/yolox/data/videos/vlc-camera10.mp4
gst-launch-1.0 \
    uridecodebin uri=$URI source::latency=200 ! nvvideoconvert ! capsfilter caps="video/x-raw(memory:NVMM),format=RGBA,width=1280,height=720" ! m.sink_0 \
    nvstreammux name=m batch-size=1 width=1920 height=1080 \
    ! nvinfer config-file-path="/research/object_detection/common/yolox/deepstream/configs/infer_x86.txt" ! yoloxparser ! nvbytetrack track-thresh=0.5 high-thresh=0.6 ! nvmultistreamtiler rows=1 columns=1 width=1280 height=720 ! nvvideoconvert ! nvdsosd ! fpsdisplaysink sync=false async=true -v | grep last-message
```

URI=file:///research/object_detection/common/yolox/data/Sgpmc065-10.mp4
URI=file:///research/object_detection/common/yolox/data/3.mp4
gst-launch-1.0 \
    uridecodebin uri=$URI source::latency=200 ! nvvideoconvert ! capsfilter caps="video/x-raw(memory:NVMM),format=RGBA,width=1280,height=1280" ! m.sink_0 \
    nvstreammux name=m batch-size=1 width=1080 height=1080 \
    ! nvinfer config-file-path="/research/object_detection/common/yolox_original/deepstream/configs/infer_rotate_x86.txt" ! yoloxparser muxer-width=1080 muxer-height=1080 threshold=0.5 ! nvbytetrack track-thresh=0.5 high-thresh=0.6 ! nvmultistreamtiler rows=1 columns=1 width=1280 height=1280 ! nvvideoconvert ! nvdsosd ! fpsdisplaysink sync=false async=true -v | grep last-message

# Metadata

Bao gồm các metadata sau trong trường hợp fisheye

* full_bbox: bbox đầy đủ, mở rộng đến mức tối đa
* foot_bbox (**default**): bbox giữ điểm chân dưới

**Chú ý**: khi sử dụng metadata cần include file *yolox_meta.h*.