//
// Created by xingyu on 1/10/23.
//
#include "cuda_utils.h"
#include <stdio.h>
/**
 *
 * @param B
 * @param N
 * @param M
 * @param faces (B, M, 3)
 * @param face_normals (B, M, 3)
 * @param vertex_normals (B, N, 3)
 */
__global__ void compute_vertex_normal_kernel(int B, int N, int M, const int *faces, const float* face_normals, float* vertex_normals) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            for (int i = 0; i < 3; ++i) {
                int v = faces[(b * M + m) * 3 + i];
                if (v != -1) {
                    atomicAdd(vertex_normals + (b * N + v) * 3, face_normals[(b * M + m) * 3]);
                    atomicAdd(vertex_normals + (b * N + v) * 3 + 1, face_normals[(b * M + m) * 3 + 1]);
                    atomicAdd(vertex_normals + (b * N + v) * 3 + 2, face_normals[(b * M + m) * 3 + 2]);
                }
            }
        }
    }
}

void compute_vertex_normal_kernel_wrapper(int B, int N, int M, const int *faces, const float* face_normals, float* vertex_normals) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    compute_vertex_normal_kernel<<<B, opt_n_threads(M), 0, stream>>>(
            B, N, M, faces, face_normals, vertex_normals);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param faces (B, M, 3)
 * @param vertex_normals (B, N, 3)
 * @param face_normals (B, M, 3)
 */
__global__ void compute_face_normal_grad_kernel(int B, int N, int M, const int *faces, const float *vertex_normal_grad, float *face_normal_grad) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            for (int i = 0; i < 3; ++i) {
                int v = faces[(b * M + m) * 3 + i];
                if (v != -1) {
                    face_normal_grad[(b * M + m) * 3] += vertex_normal_grad[(b * N + v) * 3];
                    face_normal_grad[(b * M + m) * 3 + 1] += vertex_normal_grad[(b * N + v) * 3 + 1];
                    face_normal_grad[(b * M + m) * 3 + 2] += vertex_normal_grad[(b * N + v) * 3 + 2];
                }
            }
        }
    }
}

void compute_face_normal_grad_kernel_wrapper(int B, int N, int M, const int *faces, const float *vertex_normal_grad, float *face_normal_grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    compute_face_normal_grad_kernel<<<B, opt_n_threads(M), 0, stream>>>(
            B, N, M, faces, vertex_normal_grad, face_normal_grad);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param faces (B, M, 3)
 * @param e (B, M, 3, 3)
 * @param vertex_grad (B, N, 3)
 */
__global__ void compute_vertex_normal_grad_kernel(int B, int N, int M, const int *faces, const float *e, float* vertex_grad) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            for (int i = 0; i < 3; ++i) {
                int v = faces[(b * M + m) * 3 + i];
                if (v != -1) {
                    atomicAdd(vertex_grad + (b * N + v) * 3 + 0, e[((b * M + m) * 3 + i) * 3 + 0]);
                    atomicAdd(vertex_grad + (b * N + v) * 3 + 1, e[((b * M + m) * 3 + i) * 3 + 1]);
                    atomicAdd(vertex_grad + (b * N + v) * 3 + 2, e[((b * M + m) * 3 + i) * 3 + 2]);
                }
            }
        }
    }
}

void compute_vertex_normal_grad_kernel_wrapper(int B, int N, int M, const int *faces, const float *e,
                                               float* vertex_grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    compute_vertex_normal_grad_kernel<<<B, opt_n_threads(M), 0, stream>>>(
            B, N, M, faces, e, vertex_grad);

    CUDA_CHECK_ERRORS();
}