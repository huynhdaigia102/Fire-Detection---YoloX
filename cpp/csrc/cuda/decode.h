#pragma once

#include <vector>

namespace rapid {
namespace cuda {

int decode(int batch_size,
    const void *const *inputs, void *const *outputs,
    int top_n, size_t f_width, size_t f_height, float score_thresh,
    int stride,
    void *workspace, size_t workspace_size, cudaStream_t stream);

int decode_rotate(int batch_size,
    const void *const *inputs, void *const *outputs,
    int top_n, size_t f_width, size_t f_height, float score_thresh,
    int stride,
    void *workspace, size_t workspace_size, cudaStream_t stream);
}
}