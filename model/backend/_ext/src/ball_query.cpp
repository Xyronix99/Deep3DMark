// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"
#include "cuda_utils.h"


void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

void k_neighbor_query_kernel_wrapper(int b, int n, int m,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx, float *tmp_v);

void graph_neighbor_query_kernel_wrapper(int B, int N, int M, int nsample,
                                         const int *__restrict__ header,
                                         const int *__restrict__ cnt,
                                         const int *__restrict__ v,
                                         const int *__restrict__ next,
                                         int *__restrict__ idx, int *__restrict__ num);

void get_maximum_degree_kernel_wrapper(int B, int N, int M,
                                       const int *__restrict__ header,
                                       const int *__restrict__ cnt,
                                       const int *__restrict__ v,
                                       const int *__restrict__ next,
                                       int *__restrict__ tot);


/**
 *
 * @param new_xyz (B, M, 3) selected grouping center
 * @param xyz     (B, N, 3)
 * @param radius
 * @param nsample group_size
*/
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);
    CHECK_CUDA(new_xyz);
    CHECK_CUDA(xyz);

    at::cuda::CUDAGuard device_guard(xyz.device());

    at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());

    return idx;
}

/**
 *
 * @param new_xyz (B, M, 3)
 * @param xyz     (B, N, 3)
 * @param nsample
 * @return idx    (B, M, nsample)
 */
at::Tensor k_neighbor_query(at::Tensor new_xyz, at::Tensor xyz, const int nsample) {
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);
    CHECK_CUDA(new_xyz);
    CHECK_CUDA(xyz);

    TORCH_CHECK(nsample>0, "nsample must greater than 0");
    at::cuda::CUDAGuard device_guard(xyz.device());

    at::Tensor idx = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    at::Tensor tmp_v = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));

    k_neighbor_query_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), idx.data_ptr<int>(), tmp_v.data_ptr<float>());

    return idx;
}


/**
 * Some faces are padded with -1 at the end
 * Will skip all -1 number
 * @param xyz
 * @param faces
 * @param nsample
 * @return
 */
std::vector<at::Tensor> graph_neighbor_query(at::Tensor xyz, at::Tensor faces, const int nsample) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(xyz);
    CHECK_CUDA(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    TORCH_CHECK(nsample>0, "nsample must greater than 0");
    at::cuda::CUDAGuard device_guard(xyz.device());

    const int B = xyz.size(0);
    const int N = xyz.size(1);

    auto rtn = build_graph_from_triangle(xyz, faces);
    auto header = rtn[0];
    auto cnt = rtn[1];
    auto v = rtn[2];
    auto next = rtn[3];

    const int M = v.size(1);

    at::Tensor idx = torch::full({B, N, nsample}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor num = torch::zeros({B, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    graph_neighbor_query_kernel_wrapper(B, N, M, nsample,
                                        header.data_ptr<int>(), cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>(),
                                        idx.data_ptr<int>(), num.data_ptr<int>());
    return std::vector<at::Tensor>{idx, num};
//    at::Tensor kidx = k_neighbor_query(xyz, xyz, nsample);
//    return merge_idx(idx, kidx);
}

/**
 * Some faces are padded with -1 at the end
 * Will skip all -1 number
 * @param xyz
 * @param faces
 * @return
 */
std::vector<at::Tensor> graph_neighbor_query_all(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(xyz);
    CHECK_CUDA(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    at::cuda::CUDAGuard device_guard(xyz.device());

    auto rtn = build_graph_from_triangle(xyz, faces);
    auto header = rtn[0];
    auto cnt = rtn[1];
    auto v = rtn[2];
    auto next = rtn[3];

    const int M = v.size(1);
    at::Tensor tot = torch::zeros({B, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    get_maximum_degree_kernel_wrapper(B, N, M, header.data_ptr<int>(), cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>(), tot.data_ptr<int>());
    int S = tot.max().item<int>() + 1;

    at::Tensor idx = torch::full({B, N, S}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor num = torch::zeros({B, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    graph_neighbor_query_kernel_wrapper(B, N, M, S,
                                        header.data_ptr<int>(), cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>(),
                                        idx.data_ptr<int>(), num.data_ptr<int>());
    return std::vector<at::Tensor>{idx, num};
//    at::Tensor kidx = k_neighbor_query(xyz, xyz, nsample);
//    return merge_idx(idx, kidx);
}