#include "decode.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>
#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

namespace rapid {
namespace cuda {

// copy data
__global__ void softmax_kernel(const float *data, float *scores, float *boxes, int num_elem) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx >= num_elem) return;

    boxes[idx + num_elem * 0] = data[idx + num_elem * 0];
    boxes[idx + num_elem * 1] = data[idx + num_elem * 1];
    boxes[idx + num_elem * 2] = data[idx + num_elem * 2];
    boxes[idx + num_elem * 3] = data[idx + num_elem * 3];

    scores[idx + num_elem * 0] =
        data[idx + num_elem * 4] * data[idx + num_elem * 5];

}

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return float2{v.x*invLen, v.y*invLen};
}

inline __host__ __device__ float cal_angle(const float *box, size_t f_width, size_t f_height, int stride)
{
    float2 a = float2{0, -10};
    float2 b = float2{box[0] - f_width*stride/2, box[1] - f_height*stride/2};
    float2 a_norm = normalize(a);
    float2 b_norm = normalize(b);
    float inner_product = a_norm.x*b_norm.x + a_norm.y*b_norm.y;
    float angle = acos(inner_product);
    if (b.x < 0) angle *= -1.;

    return angle;
}

int decode(int batch_size,
    const void *const *inputs, void *const *outputs,
    int top_n, size_t f_width, size_t f_height, float score_thresh,
    int stride,
    void *workspace, size_t workspace_size, cudaStream_t stream) {
    
    /**************************
    height, width: net_inshape
    num_anchors = f_height * f_width * 3
    **************************/
    
    int num_elem = f_width * f_height;

    if (!workspace || !workspace_size) {
        // scratch space size cub style
        // workspace_size  = get_size_aligned<float>(anchors.size()); // anchors
        workspace_size = get_size_aligned<bool>(num_elem);     // flags
        workspace_size += get_size_aligned<int>(num_elem);      // indices
        workspace_size += get_size_aligned<int>(num_elem);      // indices_sorted
        workspace_size += get_size_aligned<float>(num_elem);    // scores
        workspace_size += get_size_aligned<float>(num_elem);    // scores_sorted
        workspace_size += get_size_aligned<float>(num_elem);    // scores_softmax
        // workspace_size += get_size_aligned<float>(num_anchors);    // conf

        workspace_size += get_size_aligned<float>(num_elem*4);    // in_boxes

        size_t temp_size_flag = 0;
        thrust::cuda_cub::cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
        thrust::cuda_cub::cub::CountingInputIterator<int>(num_elem),
        (bool *)nullptr, (int *)nullptr, (int *)nullptr, num_elem);
        size_t temp_size_sort = 0;
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
        (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, num_elem);
        workspace_size += std::max(temp_size_flag, temp_size_sort);

        return workspace_size;
    }

    // auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
    // cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

    auto on_stream = thrust::cuda::par.on(stream);

    auto flags = get_next_ptr<bool>(num_elem, workspace, workspace_size);  // used for filtering flags by threshold
    auto indices = get_next_ptr<int>(num_elem, workspace, workspace_size);  // used for filtering index by threshold
    auto indices_sorted = get_next_ptr<int>(num_elem, workspace, workspace_size);
    auto scores = get_next_ptr<float>(num_elem, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(num_elem, workspace, workspace_size);
    auto scores_softmax = get_next_ptr<float>(num_elem, workspace, workspace_size);
    // auto conf = get_next_ptr<float>(num_anchors, workspace, workspace_size);

    auto in_boxes = get_next_ptr<float>(num_elem*4, workspace, workspace_size);

    int thread_count;
    // int num_anchor = 3;

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_data = static_cast<const float *>(inputs[0]) + batch * num_elem * 6;  // x,y,w,h,score,cls

        auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
        auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;

        // sigmoid activation
        const int thread_count_ = 1024;
        thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
        softmax_kernel<<<(num_elem + thread_count - 1) / thread_count, thread_count, 0, stream>>>(in_data, scores_softmax, in_boxes, num_elem);

        // Discard scores below threshold
        thrust::transform(on_stream, scores_softmax, scores_softmax + num_elem, flags, thrust::placeholders::_1 > score_thresh);

        int *num_selected = reinterpret_cast<int *>(indices_sorted);
        thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
            thrust::cuda_cub::cub::CountingInputIterator<int>(0),
            flags, indices, num_selected, num_elem, stream);
        cudaStreamSynchronize(stream);
        int num_detections = *thrust::device_pointer_cast(num_selected);
        
        // Only keep top n scores
        auto indices_filtered = indices;
        if (num_detections > top_n) {
            // lấy score theo indices đã chọn ở trên, sort index theo score, đẩy vào scores
            thrust::gather(on_stream, indices, indices + num_detections, scores_softmax, scores);
            // sort các giá trị trong scores đẩy vào scores_sorted để lấy n giá trị
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);
            indices_filtered = indices_sorted;
            num_detections = top_n;
        }

        // Gather boxes
        // bool has_anchors = !anchors.empty();
        thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes)),
            [=] __device__ (int i) {
                int x = i % f_width;
                int y = (i / f_width) % f_height;
                // int a = (i / 1 / f_height / f_width) % num_anchor;

                float xywh[4];  // cx,cy,w,h
                xywh[0] = in_boxes[(0 * f_height + y) * f_width + x];
                xywh[1] = in_boxes[(1 * f_height + y) * f_width + x];
                xywh[2] = in_boxes[(2 * f_height + y) * f_width + x];
                xywh[3] = in_boxes[(3 * f_height + y) * f_width + x];

                // cx,cy,w,h
                xywh[0] = (xywh[0] + x) * stride;
                xywh[1] = (xywh[1] + y) * stride;
                xywh[2] = expf(xywh[2]) * stride;
                xywh[3] = expf(xywh[3]) * stride;

                // x1,y1,x2,y2
                float4 box = float4{
                    xywh[0] - xywh[2]*0.5f,
                    xywh[1] - xywh[3]*0.5f,
                    xywh[0] + xywh[2]*0.5f,
                    xywh[1] + xywh[3]*0.5f,
                };

                return thrust::make_tuple(scores_softmax[i], box);
            });

        // Zero-out unused scores
        if (num_detections < top_n) {
            thrust::fill(on_stream, out_scores + num_detections,
                out_scores + top_n, 0.0f);
        }
    }

    return 0;

}

int decode_rotate(int batch_size,
    const void *const *inputs, void *const *outputs,
    int top_n, size_t f_width, size_t f_height, float score_thresh,
    int stride,
    void *workspace, size_t workspace_size, cudaStream_t stream) {
    
    /**************************
    height, width: net_inshape
    num_anchors = f_height * f_width * 3
    **************************/
    
    int num_elem = f_width * f_height;

    if (!workspace || !workspace_size) {
        // scratch space size cub style
        // workspace_size  = get_size_aligned<float>(anchors.size()); // anchors
        workspace_size = get_size_aligned<bool>(num_elem);     // flags
        workspace_size += get_size_aligned<int>(num_elem);      // indices
        workspace_size += get_size_aligned<int>(num_elem);      // indices_sorted
        workspace_size += get_size_aligned<float>(num_elem);    // scores
        workspace_size += get_size_aligned<float>(num_elem);    // scores_sorted
        workspace_size += get_size_aligned<float>(num_elem);    // scores_softmax
        // workspace_size += get_size_aligned<float>(num_anchors);    // conf

        workspace_size += get_size_aligned<float>(num_elem*4);    // in_boxes

        size_t temp_size_flag = 0;
        thrust::cuda_cub::cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
        thrust::cuda_cub::cub::CountingInputIterator<int>(num_elem),
        (bool *)nullptr, (int *)nullptr, (int *)nullptr, num_elem);
        size_t temp_size_sort = 0;
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
        (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, num_elem);
        workspace_size += std::max(temp_size_flag, temp_size_sort);

        return workspace_size;
    }

    // auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
    // cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

    auto on_stream = thrust::cuda::par.on(stream);

    auto flags = get_next_ptr<bool>(num_elem, workspace, workspace_size);  // used for filtering flags by threshold
    auto indices = get_next_ptr<int>(num_elem, workspace, workspace_size);  // used for filtering index by threshold
    auto indices_sorted = get_next_ptr<int>(num_elem, workspace, workspace_size);
    auto scores = get_next_ptr<float>(num_elem, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(num_elem, workspace, workspace_size);
    auto scores_softmax = get_next_ptr<float>(num_elem, workspace, workspace_size);
    // auto conf = get_next_ptr<float>(num_anchors, workspace, workspace_size);

    auto in_boxes = get_next_ptr<float>(num_elem*4, workspace, workspace_size);

    int thread_count;
    // int num_anchor = 3;

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_data = static_cast<const float *>(inputs[0]) + batch * num_elem * 6;  // x,y,w,h,score,cls

        auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
        auto out_boxes = static_cast<float5 *>(outputs[1]) + batch * top_n;

        // sigmoid activation
        const int thread_count_ = 1024;
        thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
        softmax_kernel<<<(num_elem + thread_count - 1) / thread_count, thread_count, 0, stream>>>(in_data, scores_softmax, in_boxes, num_elem);

        // Discard scores below threshold
        thrust::transform(on_stream, scores_softmax, scores_softmax + num_elem, flags, thrust::placeholders::_1 > score_thresh);

        int *num_selected = reinterpret_cast<int *>(indices_sorted);
        thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
            thrust::cuda_cub::cub::CountingInputIterator<int>(0),
            flags, indices, num_selected, num_elem, stream);
        cudaStreamSynchronize(stream);
        int num_detections = *thrust::device_pointer_cast(num_selected);
        
        // Only keep top n scores
        auto indices_filtered = indices;
        if (num_detections > top_n) {
            // lấy score theo indices đã chọn ở trên, sort index theo score, đẩy vào scores
            thrust::gather(on_stream, indices, indices + num_detections, scores_softmax, scores);
            // sort các giá trị trong scores đẩy vào scores_sorted để lấy n giá trị
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);
            indices_filtered = indices_sorted;
            num_detections = top_n;
        }

        // Gather boxes
        // bool has_anchors = !anchors.empty();
        thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes)),
            [=] __device__ (int i) {
                int x = i % f_width;
                int y = (i / f_width) % f_height;
                // int a = (i / 1 / f_height / f_width) % num_anchor;

                float xywh[4];  // cx,cy,w,h
                xywh[0] = in_boxes[(0 * f_height + y) * f_width + x];
                xywh[1] = in_boxes[(1 * f_height + y) * f_width + x];
                xywh[2] = in_boxes[(2 * f_height + y) * f_width + x];
                xywh[3] = in_boxes[(3 * f_height + y) * f_width + x];

                // cx,cy,w,h
                xywh[0] = (xywh[0] + x) * stride;
                xywh[1] = (xywh[1] + y) * stride;
                xywh[2] = expf(xywh[2]) * stride;
                xywh[3] = expf(xywh[3]) * stride;

                // cal angle
                float angle = cal_angle(xywh, f_width, f_height, stride);

                // x1,y1,x2,y2
                float5 box = float5{
                    xywh[0],
                    xywh[1],
                    xywh[2],
                    xywh[3],
                    angle
                };

                return thrust::make_tuple(scores_softmax[i], box);
            });

        // Zero-out unused scores
        if (num_detections < top_n) {
            thrust::fill(on_stream, out_scores + num_detections,
                out_scores + top_n, 0.0f);
        }
    }

    return 0;

}

}
}
