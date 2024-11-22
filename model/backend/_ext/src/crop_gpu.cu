//
// Created by xingyu on 11/29/22.
//


#include "cuda_utils.h"
#include "utils.h"

void bitonic_sort(int *values);

/**
 *
 * @param R
 * @param S
 * @param cropped_idx (B, S)
 * @param is_remain (B, N)
 * @param tmp (B)
 * @param remain_idx (B, R)
 */
__global__ void crop_points_kernel(const int B, const int R, const int S,
                                   const int *__restrict__ cropped_idx,
                                   bool *__restrict__ is_remain, int *__restrict__ tmp,
                                   int *__restrict__ remain_idx) {
    const int N = R + S;
    int batch_index = blockIdx.x;
    cropped_idx += batch_index * S;
    is_remain += batch_index * N;
    remain_idx += batch_index * R;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < S; j += stride) {
        int idx = cropped_idx[j];
        is_remain[idx] = false;
    }
    // MUST WAIT UNTIL ALL THREADS FINISH
    __syncthreads();

    for (int j = index; j < N; j += stride) {
        if (is_remain[j]) {
            int cnt = atomicAdd(tmp + batch_index, 1);
            remain_idx[cnt] = j;
        }
    }
}


void crop_points_kernel_wrapper(const int B, const int R, const int S,
                                const int *__restrict__ cropped_idx,
                                bool *__restrict__ is_remain,
                                int *__restrict__ tmp,
                                int *__restrict__ remain_idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    crop_points_kernel<<<B, opt_n_threads(R+S), 0, stream>>>(
            B, R, S, cropped_idx, is_remain, tmp, remain_idx);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param S
 * @param M
 * @param faces (B, M, 3)
 * @param cropped_idx (B, S)
 * @param is_remain (B, M)
 * @param tmp (B)
 * @param cropped_faces (B, M, 3)
 */
__global__ void crop_faces_kernel(const int B, const int S, const int M,
                                  const int *__restrict__ faces,
                                  const int *__restrict__ cropped_idx,
                                  bool *__restrict__ is_remain,
                                  int *__restrict__ tmp,
                                  int *__restrict__ cropped_faces) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) { // batch_index
        for (int m = blockIdx.y; m < M; m += gridDim.y) { // face_index
            for (int s = threadIdx.x; s < S; s += blockDim.x) { // crop_index
                int faces_1 = faces[(b * M + m) * 3];
                int faces_2 = faces[(b * M + m) * 3 + 1];
                int faces_3 = faces[(b * M + m) * 3 + 2];
                int idx = cropped_idx[b * S + s];
                if (faces_1 == idx || faces_2 == idx || faces_3 == idx) {
                    is_remain[b * M + m] = false;
                }
            }
        }
    }
    // THREAD 0 MUST WAIT UNTIL ALL THREADS FINISH
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int m = blockIdx.y; m < M; m += gridDim.y) {
                if (is_remain[b * M + m]) {
                    int cnt = atomicAdd(tmp + b, 1);
                    cropped_faces[(b * M + cnt) * 3] = faces[(b * M + m) * 3];
                    cropped_faces[(b * M + cnt) * 3 + 1] = faces[(b * M + m) * 3 + 1];
                    cropped_faces[(b * M + cnt) * 3 + 2] = faces[(b * M + m) * 3 + 2];
                }
            }
        }
    }
}

void crop_faces_kernel_wrapper(const int B, const int S, const int M,
                               const int *__restrict__ faces,
                               const int *__restrict__ cropped_idx,
                               bool *__restrict__ is_remain,
                               int *__restrict__ tmp,
                               int *__restrict__ cropped_faces) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    crop_faces_kernel<<<opt_block_config(B, M), opt_n_threads(S), 0, stream>>>(B, S, M,
                      faces, cropped_idx, is_remain, tmp, cropped_faces);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param R
 * @param M
 * @param remain_idx (B, R)
 * @param cropped_faces (B, M, 3)
 */
__global__ void reorder_cropped_faces_kernel(const int B, const int R, const int M,
                                             int *__restrict__ remain_idx,
                                             int *__restrict__ cropped_faces) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {// batch_idx
        for (int m = blockIdx.y; m < M; m += gridDim.y) { // face_idx
            for (int r = threadIdx.x; r < R; r += blockDim.x) { // remain_idx
                int idx = remain_idx[b * R + r];
                for (int i = 0; i < 3; ++i) {
                    if (cropped_faces[(b * M + m) * 3 + i] == idx) {
                        cropped_faces[(b * M + m) * 3 + i] = r;
                    }
                }

            }
        }
    }
}

/**
 *
 * @param B
 * @param R
 * @param M
 * @param remain_idx (B, R)
 * @param tmp (B, R)
 * @param croppped_faces (B, M, 3)
 */
void reorder_cropped_faces_kernel_wrapper(const int B, const int R, const int M,
                                          int *__restrict__ remain_idx,
                                          int *__restrict__ cropped_faces) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    reorder_cropped_faces_kernel<<<opt_block_config(B, M), opt_n_threads(R), 0, stream>>>(B, R, M, remain_idx, cropped_faces);

    CUDA_CHECK_ERRORS();
}

