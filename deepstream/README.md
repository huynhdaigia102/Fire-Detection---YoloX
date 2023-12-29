# Build app + lib

```bash
mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-5.0 \
    -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" \
    -DPLATFORM_TEGRA=ON ..
make
```


# Run app

./detect rtsp://admin:meditech123@192.168.100.40:554/
