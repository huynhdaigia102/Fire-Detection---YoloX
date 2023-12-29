#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/cuda/nms.h"
#include "csrc/cuda/nms_rotated.h"
#include "csrc/cuda/decode.h"
#include "engine.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

vector<at::Tensor> nms(at::Tensor scores, at::Tensor boxes, float nms_thresh, int detections_per_im) {
    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();
    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());

    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr()};

    // Create scratch buffer
    int size = rapid::cuda::nms(batch, nullptr, nullptr, count,
        detections_per_im, nms_thresh, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Perform NMS
    rapid::cuda::nms(batch, inputs.data(), outputs.data(), count, detections_per_im, 
        nms_thresh, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    
    return {nms_scores, nms_boxes};
}

vector<at::Tensor> decode(at::Tensor data, int top_n, float score_thresh, int stride) {
    
    CHECK_INPUT(data);

    int batch = data.size(0);
    int f_height = data.size(2);
    int f_width = data.size(3);
    auto options = data.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, 4}, options);  // 4=(cx,cy,w,h)

    vector<void *> inputs = {data.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr()};

    // Create scratch buffer
    int size = rapid::cuda::decode(batch, nullptr, nullptr,
        top_n, f_width, f_height, score_thresh,
        stride,
        nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Decode boxes
    rapid::cuda::decode(batch, inputs.data(), outputs.data(),
        top_n, f_width, f_height, score_thresh,
        stride,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {scores, boxes};
}

vector<at::Tensor> decode_rotate(at::Tensor data, int top_n, float score_thresh, int stride) {
    
    CHECK_INPUT(data);

    int batch = data.size(0);
    int f_height = data.size(2);
    int f_width = data.size(3);
    auto options = data.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, 5}, options);  // 4=(cx,cy,w,h)

    vector<void *> inputs = {data.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr()};

    // Create scratch buffer
    int size = rapid::cuda::decode_rotate(batch, nullptr, nullptr,
        top_n, f_width, f_height, score_thresh,
        stride,
        nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Decode boxes
    rapid::cuda::decode_rotate(batch, inputs.data(), outputs.data(),
        top_n, f_width, f_height, score_thresh,
        stride,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {scores, boxes};
}


vector<at::Tensor> nms_rotated(at::Tensor scores, at::Tensor boxes, float nms_thresh, int detections_per_im) {
    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();
    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, 5}, boxes.options());

    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr()};

    // Create scratch buffer
    int size = rapid::cuda::nms_rotated(batch, nullptr, nullptr, count,
        detections_per_im, nms_thresh, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Perform NMS
    rapid::cuda::nms_rotated(batch, inputs.data(), outputs.data(), count, detections_per_im, 
        nms_thresh, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    
    return {nms_scores, nms_boxes};
}


vector<at::Tensor> infer(rapid::Engine &engine, at::Tensor data) {
    CHECK_INPUT(data);

    int batch = data.size(0);
    // auto output = at::zeros({batch, 6, 2835}, data.options());
    int num_detections = engine.getMaxDetections();
    auto scores = at::zeros({batch, num_detections}, data.options());
    auto boxes = at::zeros({batch, num_detections, 5}, data.options());

    // auto output_1 = at::zeros({batch, 6, 60, 60}, data.options());
    // auto output_2 = at::zeros({batch, 6, 30, 30}, data.options());
    // auto output_3 = at::zeros({batch, 6, 15, 15}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, scores, boxes}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {scores, boxes};
}

vector<at::Tensor> infer_rotate(rapid::EngineRotate &engine, at::Tensor data) {
    CHECK_INPUT(data);

    int batch = data.size(0);
    // auto output = at::zeros({batch, 6, 2835}, data.options());
    int num_detections = engine.getMaxDetections();
    auto scores = at::zeros({batch, num_detections}, data.options());
    auto boxes = at::zeros({batch, num_detections, 5}, data.options());

    // auto output_1 = at::zeros({batch, 6, 60, 60}, data.options());
    // auto output_2 = at::zeros({batch, 6, 30, 30}, data.options());
    // auto output_3 = at::zeros({batch, 6, 15, 15}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, scores, boxes}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {scores, boxes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<rapid::Engine>(m, "Engine")
        .def(pybind11::init<const char *, size_t, const vector<int>&,
            float, int, const vector<float>&,
            float, int,
            bool>())
        .def("save", &rapid::Engine::save)
        .def("infer", &rapid::Engine::infer)
        // .def_property_readonly("input_size", &sample_onnx::Engine::getInputSize)
        .def_static("load", [](const string &path) {
            return new rapid::Engine(path);
        })
        .def("__call__", [](rapid::Engine &engine, at::Tensor data) {
            return infer(engine, data);
        });
    
    pybind11::class_<rapid::EngineRotate>(m, "EngineRotate")
        .def(pybind11::init<const char *, size_t, const vector<int>&,
            float, int, const vector<float>&,
            float, int,
            bool>())
        .def("save", &rapid::EngineRotate::save)
        .def("infer", &rapid::EngineRotate::infer)
        .def_static("load", [](const string &path) {
            return new rapid::EngineRotate(path);
        })
        .def("__call__", [](rapid::EngineRotate &engine, at::Tensor data) {
            return infer_rotate(engine, data);
        });
    m.def("decode", &decode);
    m.def("nms", &nms);
    m.def("decode_rotate", &decode_rotate);
    m.def("nms_rotated", &nms_rotated);
}