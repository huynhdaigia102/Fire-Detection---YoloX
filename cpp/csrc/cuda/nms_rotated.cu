#include "nms_rotated.h"
#include "utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>


namespace rapid {
namespace cuda {

__global__ void nms_rotated_kernel(
    const int num_per_thread, const float threshold, const int num_detections,
    const int *indices, float *scores, const float5 *boxes) {
    
    // Go through detections by descending score
    for (int m = 0; m < num_detections; m++) {
        for (int n = 0; n < num_per_thread; n++) {
            int i = threadIdx.x * num_per_thread + n;
            if (i < num_detections && m < i && scores[m] > 0.0f) {
                int idx = indices[i];
                int max_idx = indices[m];

                float5 box1_raw = boxes[idx];
                float5 box2_raw = boxes[max_idx];

                RotatedBox<float> box1, box2;
                auto center_shift_x = (box1_raw.x + box2_raw.x) / 2.0;
                auto center_shift_y = (box1_raw.y + box2_raw.y) / 2.0;
                box1.x_ctr = box1_raw.x - center_shift_x;
                box1.y_ctr = box1_raw.y  - center_shift_y;
                box1.w = box1_raw.w;
                box1.h = box1_raw.h;
                box1.a = box1_raw.a;
                box2.x_ctr = box2_raw.x - center_shift_x;
                box2.y_ctr = box2_raw.y - center_shift_y;
                box2.w = box2_raw.w;
                box2.h = box2_raw.h;
                box2.a = box2_raw.a;

                float area1 = box1.w * box1.h;
                float area2 = box2.w * box2.h;
                if (area1 < 1e-14 || area2 < 1e-14) {
                    scores[i] = 0.0f;
                    continue;
                }

                float intersection = rotated_boxes_intersection<float>(box1, box2);
                float overlap = intersection / (area1 + area2 - intersection);
                // float overlap = 0.5f;

                if (overlap > threshold) {
                    scores[i] = 0.0f;
                }
            }
        }
        // Sync discarded detections
        __syncthreads();
    }
}

int nms_rotated(int batch_size,
    const void *const *inputs, void *const *outputs,
    size_t count, int detections_per_im, float nms_thresh,
    void *workspace, size_t workspace_size, cudaStream_t stream) {
    
    if (!workspace || !workspace_size) {
        // Return required scratch space size cub style
        workspace_size  = get_size_aligned<bool>(count);  // flags
        workspace_size += get_size_aligned<int>(count);   // indices
        workspace_size += get_size_aligned<int>(count);   // indices_sorted
        workspace_size += get_size_aligned<float>(count); // scores
        workspace_size += get_size_aligned<float>(count); // scores_sorted
        
        size_t temp_size_flag = 0;
        thrust::cuda_cub::cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
            thrust::cuda_cub::cub::CountingInputIterator<int>(count),
            (bool *)nullptr, (int *)nullptr, (int *)nullptr, count);
        size_t temp_size_sort = 0;
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
            (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, count);
        workspace_size += std::max(temp_size_flag, temp_size_sort);
    
        return workspace_size;
    }

    auto on_stream = thrust::cuda::par.on(stream);

    auto flags = get_next_ptr<bool>(count, workspace, workspace_size);
    auto indices = get_next_ptr<int>(count, workspace, workspace_size);
    auto indices_sorted = get_next_ptr<int>(count, workspace, workspace_size);
    auto scores = get_next_ptr<float>(count, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(count, workspace, workspace_size);

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_scores = static_cast<const float *>(inputs[0]) + batch * count;
        auto in_boxes = static_cast<const float5 *>(inputs[1]) + batch * count;
        
        auto out_scores = static_cast<float *>(outputs[0]) + batch * detections_per_im;
        auto out_boxes = static_cast<float5 *>(outputs[1]) + batch * detections_per_im;
    
        // Discard null scores
        thrust::transform(on_stream, in_scores, in_scores + count,
            flags, thrust::placeholders::_1 > 0.0f);
        
        int *num_selected = reinterpret_cast<int *>(indices_sorted);
        thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size, thrust::cuda_cub::cub::CountingInputIterator<int>(0),
            flags, indices, num_selected, count, stream);
        cudaStreamSynchronize(stream);
        int num_detections = *thrust::device_pointer_cast(num_selected);

        // Sort scores and corresponding indices
        thrust::gather(on_stream, indices, indices + num_detections, in_scores, scores);
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);

        // Launch actual NMS kernel - 1 block with each thread handling n detections
        const int max_threads = 256;
        int num_per_thread = ceil((float)num_detections / max_threads);
        nms_rotated_kernel<<<1, max_threads, 0, stream>>>(num_per_thread, nms_thresh, num_detections,
            indices_sorted, scores_sorted, in_boxes);

        // Re-sort with updated scores
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            scores_sorted, scores, indices_sorted, indices, num_detections, 0, sizeof(*scores)*8, stream);
        
        // Gather filtered scores, boxes
        num_detections = min(detections_per_im, num_detections);
        cudaMemcpyAsync(out_scores, scores, num_detections * sizeof *scores, cudaMemcpyDeviceToDevice, stream);
        if (num_detections < detections_per_im) {
            thrust::fill_n(on_stream, out_scores + num_detections, detections_per_im - num_detections, 0);
        }
        thrust::gather(on_stream, indices, indices + num_detections, in_boxes, out_boxes);
    }
    return 0;
}

}
}