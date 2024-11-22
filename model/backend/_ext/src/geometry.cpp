//
// Created by xingyu on 1/10/23.
//

#include "geometry.h"
#include "utils.h"
#include "group_points.h"
#include "cuda_utils.h"

namespace ti = torch::indexing;
namespace F = torch::nn::functional;

void compute_vertex_normal_kernel_wrapper(int B, int N, int M, const int *faces, const float* face_normals, float* vertex_normals);

void compute_vertex_normal_grad_kernel_wrapper(int B, int N, int M, const int *faces, const float *e,
                                               float* grad);

void compute_face_normal_grad_kernel_wrapper(int B, int N, int M, const int *faces, const float *vertex_normal_grad, float *face_normal_grad);

/**
 * Faces are padded with -1
 * @param xyz (B, N, 3)
 * @param faces (B, M, 3)
 * @return
 */
std::vector<at::Tensor> compute_vertex_normals(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(xyz);
    CHECK_CUDA(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = faces.size(1);
    at::cuda::CUDAGuard device_guard(xyz.device());

    at::Tensor vertex_normals = torch::zeros({B, N, 3}, at::device(xyz.device()).dtype(at::ScalarType::Float));
//    at::Tensor face_normals = torch::zeros({B, M, 3}, at::device(xyz.device()).dtype(at::ScalarType::Float));

    at::Tensor vertex_per_face = group_points(xyz.permute({0, 2, 1}).contiguous(), faces).permute({0, 2, 3, 1}); // (B, M, 3, 3)

    auto v2 = vertex_per_face.index({ti::Slice(), ti::Slice(), 2, ti::Slice()});
    auto v1 = vertex_per_face.index({ti::Slice(), ti::Slice(), 1, ti::Slice()});
    auto v0 = vertex_per_face.index({ti::Slice(), ti::Slice(), 0, ti::Slice()});
    auto vec1 = v2.subtract(v1);  // (B, M, 3)
    auto vec2 = v0.subtract(v1);
    at::Tensor face_normals = torch::cross(vec1, vec2, -1); // (B, M, 3)
//    face_normals = F::normalize(face_normals, F::NormalizeFuncOptions().dim(-1));
    compute_vertex_normal_kernel_wrapper(B, N, M,
                                         faces.data_ptr<int>(), face_normals.data_ptr<float>(),
                                         vertex_normals.data_ptr<float>()
                                         );
//    face_normals = F::normalize(face_normals, F::NormalizeFuncOptions().dim(-1));
//    DEBUG(vertex_normals[0][0]);
    double eps = 1e-12;
    at::Tensor norm = vertex_normals.norm(2, -1).clamp_min(eps);
//    DEBUG(norm[0][0]);
    return std::vector<at::Tensor>{vertex_normals/norm.view({B, N, 1}).expand_as(vertex_normals), vertex_normals, norm};
}

/**
 * v/norm
 *
 *
 *
 *
 *
 *
 * @param grad_out (B, N, 3)
 * @param xyz (B, N, 3)
 * @param faces (B, M, 3)
 * @param magtitude (B, N, 1)
 * @return
 */
at::Tensor compute_vertex_normals_grad(at::Tensor vertex_normal_grad, at::Tensor xyz, at::Tensor faces, at::Tensor normals, at::Tensor norm) {
    CHECK_CONTIGUOUS(vertex_normal_grad);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(vertex_normal_grad);
    CHECK_CUDA(xyz);
    CHECK_CUDA(faces);
    CHECK_IS_FLOAT(vertex_normal_grad);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int B = vertex_normal_grad.size(0);
    const int N = vertex_normal_grad.size(1);
    const int M = faces.size(1);
    double eps = 1e-12;
    at::cuda::CUDAGuard device_guard(xyz.device());

//    DEBUG(vertex_normal_grad[0][0]);
    norm = norm.clamp_min(eps).view({B, N, 1}).expand_as(normals);
    auto vertex_normal_grad_before_normalize_1  = vertex_normal_grad / norm; // get vertex_normal_grad before normalize part 1
//    DEBUG(vertex_normal_grad_before_normalize_1[0][0]);
    auto norm_grad = vertex_normal_grad * (-normals) / norm / norm; // get norm grad
//    DEBUG(norm_grad[0][0]);
    auto vertex_normal_grad_before_normalize_2 = norm_grad.sum(-1).view({B, N, 1}).expand_as(normals) * normals / norm;
//    DEBUG(vertex_normal_grad_before_normalize_2[0][0]);

    auto vertex_normal_grad_before_normalize = vertex_normal_grad_before_normalize_1 + vertex_normal_grad_before_normalize_2;
//    DEBUG(vertex_normal_grad_before_normalize[0][0]);

    at::Tensor vertex_grad = torch::zeros({B, N, 3}, at::device(vertex_normal_grad.device()).dtype(at::ScalarType::Float));
    at::Tensor face_normal_grad = torch::zeros({B, M, 3}, at::device(vertex_normal_grad.device()).dtype(at::ScalarType::Float));

    compute_face_normal_grad_kernel_wrapper(B, N, M, faces.data_ptr<int>(), vertex_normal_grad_before_normalize.data_ptr<float>(), face_normal_grad.data_ptr<float>());
//    DEBUG(face_normal_grad[0].index({ti::Slice(0, 9)}));

    at::Tensor vertex_per_face = group_points(xyz.permute({0, 2, 1}).contiguous(), faces).permute({0, 2, 3, 1}); // (B, M, 3, 3)

    at::Tensor v2 = vertex_per_face.index({ti::Slice(), ti::Slice(), 2, ti::Slice()}); // (B, M, 3)
    at::Tensor v1 = vertex_per_face.index({ti::Slice(), ti::Slice(), 1, ti::Slice()});
    at::Tensor v0 = vertex_per_face.index({ti::Slice(), ti::Slice(), 0, ti::Slice()});
    auto vec1 = v2.subtract(v1); // (B, M, 3)
    auto vec2 = v0.subtract(v1);

    auto v2_grad = torch::cross(vec2, face_normal_grad);
    auto v0_grad = torch::cross(face_normal_grad, vec1);
    auto v1_grad = - v2_grad - v0_grad;

//    DEBUG(v2_grad[0].index({ti::Slice(0, 9)})); // (B, M, 3)
//    DEBUG(v1_grad[0].index({ti::Slice(0, 9)}));
//    DEBUG(v0_grad[0].index({ti::Slice(0, 9)}));

    auto e = torch::stack({v0_grad, v1_grad, v2_grad}, 2);

//    DEBUG(e[0][1][0] + e[0][0][2] + e[0][2][2] + e[0][5][2] + e[0][6][2] + e[0][7][2] + e[0][8][2]);

    compute_vertex_normal_grad_kernel_wrapper(B, N, M,
                                              faces.data_ptr<int>(), e.data_ptr<float>(),
                                              vertex_grad.data_ptr<float>());

    return vertex_grad;
}