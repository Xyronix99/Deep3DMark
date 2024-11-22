//
// Created by xingyu on 12/14/22.
//
#include "cuda_utils.h"
#include "utils.h"

void build_graph_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ polygons,
                                              int *__restrict__ header,
                                              int *__restrict__ cnt,
                                              int *__restrict__ v,
                                              int *__restrict__ next);

void build_edge_matrix_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ faces, int *__restrict__ out);

void build_graph_from_img_kernel_wrapper(int B, int H, int W, const float *__restrict__ img, float *__restrict__ xyz, int *__restrict__ idx);

/**
 * Some polygons are padded with -1 in the end
 * Will Skip all -1
 * @param xyz
 * @param polygons
 * @return
 */
std::vector<at::Tensor> build_graph_from_triangle(at::Tensor xyz, at::Tensor polygons) {
    CHECK_CONTIGUOUS(polygons);
    CHECK_CUDA(polygons);
    CHECK_IS_INT(polygons);

    at::cuda::CUDAGuard device_guard(xyz.device());

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = polygons.size(1);

    at::Tensor header = torch::full({B, N}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor cnt = torch::zeros({B}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor v = torch::zeros({B, 6*M}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor next = torch::full({B, 6*M}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));

    build_graph_from_triangle_kernel_wrapper(B, N, M, polygons.data_ptr<int>(), header.data_ptr<int>(),
                                                  cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>());

    return std::vector<at::Tensor>{header, cnt, v, next};
}

at::Tensor build_edge_matrix_from_triangle(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(faces);
    CHECK_IS_INT(faces);

    at::cuda::CUDAGuard device_guard(xyz.device());

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = faces.size(1);

    at::Tensor out = torch::zeros({B, N, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    build_edge_matrix_from_triangle_kernel_wrapper(B, N, M, faces.data_ptr<int>(), out.data_ptr<int>());

    return out;
}

std::vector<at::Tensor> build_graph_from_img(at::Tensor img) {
    CHECK_CONTIGUOUS(img);
    CHECK_CUDA(img);
    CHECK_IS_FLOAT(img);

    at::cuda::CUDAGuard device_guard(img.device());

    const int B = img.size(0);
    const int H = img.size(2);
    const int W = img.size(3);

    at::Tensor xyz = torch::zeros({B, H * W, 3}, at::device(img.device()).dtype(at::ScalarType::Float));
    at::Tensor idx = torch::full({B, H * W, 9}, -1, at::device(img.device()).dtype(at::ScalarType::Int));

    build_graph_from_img_kernel_wrapper(B, H, W, img.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<int>());

    return std::vector<at::Tensor>{xyz, idx};
}
