#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

namespace rapid {

class Engine {
public:
    Engine(const string &engine_path, bool verbose=true);
    Engine(const char *onnx_model, size_t onnx_size, const vector<int> &dynamic_batch_opts,
    float score_thresh, int top_n, const vector<float> &strides, // for decode
    float nms_thresh, int detections_per_im,  // for nms
    bool verbose, size_t workspace_size=(1ULL << 30));

    ~Engine();

    void save(const string &path);
    void infer(vector<void *> &buffers, int batch);

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // Get max allowed batch size
    int getMaxBatchSize();

    // Get max number of detections
    int getMaxDetections();

    // Get stride
    int getStride();

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    void _load(const string &path);
    void _prepare();

};

class EngineRotate {
public:
    EngineRotate(const string &engine_path, bool verbose=true);
    EngineRotate(const char *onnx_model, size_t onnx_size, const vector<int> &dynamic_batch_opts,
    float score_thresh, int top_n, const vector<float> &strides, // for decode
    float nms_thresh, int detections_per_im,  // for nms
    bool verbose, size_t workspace_size=(1ULL << 30));

    ~EngineRotate();

    void save(const string &path);
    void infer(vector<void *> &buffers, int batch);

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // Get max allowed batch size
    int getMaxBatchSize();

    // Get max number of detections
    int getMaxDetections();

    // Get stride
    int getStride();

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    void _load(const string &path);
    void _prepare();

};

}