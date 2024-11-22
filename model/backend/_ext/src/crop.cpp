//
// Created by xingyu on 11/29/22.
//
#include "crop.h"
#include "sampling.h"
#include "ball_query.h"
#include "utils.h"
#include "cuda_utils.h"

void crop_points_kernel_wrapper(const int B, const int R, const int S,
                                const int *__restrict__ cropped_idx,
                                bool *__restrict__ is_remain,
                                int *__restrict__ tmp,
                                int *__restrict__ remain_idx);

void crop_faces_kernel_wrapper(const int B, const int S, const int M,
                               const int *__restrict__ faces,
                               const int *__restrict__ cropped_idx,
                               bool *__restrict__ is_remain,
                               int *__restrict__ tmp,
                               int *__restrict__ cropped_faces);

void reorder_cropped_faces_kernel_wrapper(const int B, const int R, const int M,
                                          int *__restrict__ remain_idx,
                                          int *__restrict__ cropped_faces);

/**
 *
 * @param xyz   (B, N, 3)
 * @param faces (B, M, 3)
 * @param ratio S = N * ratio
 * @return remain_idx, cropped_idx  (B, N-S) (B, S)
 */
std::vector<at::Tensor> random_crop(at::Tensor xyz, at::Tensor faces, const float ratio) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CUDA(xyz);
    CHECK_IS_FLOAT(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(faces);
    CHECK_IS_INT(faces);
    at::cuda::CUDAGuard device_guard(xyz.device());

    /** initialize all necessary variable*/
    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int S = N * ratio;
    const int R = N - S;
    const int M = faces.size(1);
    at::Tensor xyz_trans = xyz.permute({0, 2, 1}).contiguous();
    at::Tensor faces_trans = faces.permute({0, 2, 1}).contiguous();
    at::Tensor remain_idx = torch::zeros({B, R}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor is_remain = torch::full({B, N}, true, at::device(xyz.device()).dtype(at::ScalarType::Bool));
    at::Tensor tmp = torch::zeros({B}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    /** get the indices of points being cropped **/
    at::Tensor center_idx = torch::randint(0, N, {B, 1},
                                           at::device(xyz.device()).dtype(at::ScalarType::Int)); // (B, 1)
    at::Tensor center = gather_points(xyz_trans, center_idx).permute({0, 2, 1}).contiguous(); // (B, 1, 3)
    at::Tensor cropped_idx = k_neighbor_query(center, xyz, S).squeeze(1); // (B, S)
//    DEBUG("hi1");
    /** get the indices of points remained **/
    crop_points_kernel_wrapper(B, R, S,
                               cropped_idx.data_ptr<int>(), is_remain.data_ptr<bool>(), tmp.data_ptr<int>(), remain_idx.data_ptr<int>());
    at::Tensor cropped_xyz = gather_points(xyz_trans, remain_idx).permute({0, 2, 1}).contiguous(); // (B, R, 3)
//    DEBUG("hi2");
    tmp = torch::zeros({B}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    is_remain = torch::full({B, M}, true, at::device(xyz.device()).dtype(at::ScalarType::Bool));
    at::Tensor cropped_faces = torch::full({B, M, 3}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
//    DEBUG("hi3");
    crop_faces_kernel_wrapper(B, S, M,
                              faces.data_ptr<int>(), cropped_idx.data_ptr<int>(),
                              is_remain.data_ptr<bool>(), tmp.data_ptr<int>(), cropped_faces.data_ptr<int>());
//    DEBUG("hi4");
    reorder_cropped_faces_kernel_wrapper(B, R, M,
                                         remain_idx.data_ptr<int>(), cropped_faces.data_ptr<int>());
//    DEBUG("hi5");
    return std::vector<at::Tensor>{cropped_xyz, cropped_faces};
}