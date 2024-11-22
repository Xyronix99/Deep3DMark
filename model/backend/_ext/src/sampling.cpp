// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"
#include "cuda_utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_CUDA(points);
    CHECK_CUDA(idx);

    at::cuda::CUDAGuard device_guard(points.device());

    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data_ptr<float>(),
                                 idx.data_ptr<int>(), output.data_ptr<float>());

    return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_CUDA(idx);
    CHECK_CUDA(grad_out);

    at::cuda::CUDAGuard device_guard(grad_out.device());

    at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data_ptr<float>(),
                                      idx.data_ptr<int>(), output.data_ptr<float>());

    return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);
    CHECK_CUDA(points);

    at::cuda::CUDAGuard device_guard(points.device());

    at::Tensor output = torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

    at::Tensor tmp = torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());

    return output;
}
