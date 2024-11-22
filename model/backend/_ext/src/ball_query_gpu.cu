// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "utils.h"

__device__ bool heap_insert(const float value, const int idx,
                            const int cnt, const int max_heap_size,
                            float *__restrict__ tmp_v,
                            int *__restrict__ tmp_i);

//////////////////////////////////////////////////////////////////
////////////                ball query                /////////////
//////////////////////////////////////////////////////////////////


// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}

//////////////////////////////////////////////////////////////////
////////////                k_neighbor_query                /////////////
//////////////////////////////////////////////////////////////////

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// intermediate: tmp_v(b, m, nsample) tmp_i(b, m, nsample)
// output: idx(b, m, nsample)
__global__ void k_neighbor_query_kernel(int b, int n, int m, int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx,
                                        float *__restrict__ tmp_v){
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;
  tmp_v += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];

    // current heap top index = tmp_v + j * nsample
    // current heap top index = tmp_i + j * nsample
    for (int k = 0, cnt = 0; k < n; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float dist = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
      cnt += heap_insert(dist, k,
                  cnt, nsample,
                  tmp_v + j * nsample, idx + j * nsample);
    }
  }
}

void k_neighbor_query_kernel_wrapper(int b, int n, int m,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx, float *tmp_v) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  k_neighbor_query_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, nsample, new_xyz, xyz, idx, tmp_v);
  CUDA_CHECK_ERRORS();
}

//////////////////////////////////////////////////////////////////
////////////                topology query                /////////////
//////////////////////////////////////////////////////////////////

__device__ void bfs(int source, int nsample,
                    const int *__restrict__ header,
                    const int *__restrict__ cnt,
                    const int *__restrict__ v,
                    const int *__restrict__ next,
                    int *__restrict__ out, int *__restrict__ num) {
    out[0] = source;
    int h,t;
    for (h=0, t=1; h<t && t<nsample; ++h) {
        int x = out[h];
        for (int i = header[x]; i != -1 && t < nsample; i = next[i]) {
            int to = v[i];
            bool visited = false;
            for (int j = 0; j < t; ++j) {
                if (out[j] == to) {
                    visited = true;
                    break;
                }
            }
            if (!visited) {
                out[t++] = to;
            }
        }
        break; // 1 step
    }
    *num = t;
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param nsample
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, M)
 * @param next (B, M)
 * @param vis (B, N)
 * @param idx (B, N, nsample)
 * @param num (B, N)
 */
__global__ void graph_neighbor_query_kernel(int B, int N, int M, int nsample,
                                            const int *__restrict__ header,
                                            const int *__restrict__ cnt,
                                            const int *__restrict__ v,
                                            const int *__restrict__ next,
                                            int *__restrict__ idx, int *__restrict__ num) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            bfs(n, nsample,
                header+b*N, cnt+b, v+b*M, next+b*M,
                idx+(b*N+n)*nsample, num+b*N+n);
        }
    }
}


void graph_neighbor_query_kernel_wrapper(int B, int N, int M, int nsample,
                                         const int *__restrict__ header,
                                         const int *__restrict__ cnt,
                                         const int *__restrict__ v,
                                         const int *__restrict__ next,
                                         int *__restrict__ idx, int *__restrict__ num) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    graph_neighbor_query_kernel<<<B, opt_n_threads(N), 0, stream>>>(B, N, M, nsample,
                                                                    header, cnt, v, next, idx, num);
    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param header (B, N)
 * @param cnt (B, M)
 * @param v (B, M)
 * @param next (B, M)
 * @param tot (B, N)
 */
__global__ void get_maximum_degree_kernel(int B, int N, int M,
                                          const int *__restrict__ header,
                                          const int *__restrict__ cnt,
                                          const int *__restrict__ v,
                                          const int *__restrict__ next,
                                          int *__restrict__ tot) {
    for (int b=blockIdx.x; b<B; b+=gridDim.x) {
        for (int n=threadIdx.x; n<N; n+=blockDim.x) {
            for (int i=header[b * N + n]; i!=-1; i=next[b * M + i]) {
                int to = v[b * M + i];
                tot[b * N + n]++;
            }
        }
    }
}

void get_maximum_degree_kernel_wrapper(int B, int N, int M,
                                       const int *__restrict__ header,
                                       const int *__restrict__ cnt,
                                       const int *__restrict__ v,
                                       const int *__restrict__ next,
                                       int *__restrict__ tot) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_maximum_degree_kernel<<<B, opt_n_threads(N), 0, stream>>>(B, N, M,
                                                                    header, cnt, v, next, tot);
    CUDA_CHECK_ERRORS();
}



//////////////////////////////////////////////////////////////////
////////////                heap sort                /////////////
//////////////////////////////////////////////////////////////////

__device__ void swap(const int idx1, const int idx2,
                     float *__restrict__ tmp_v, int *__restrict__ tmp_i) {
    float idx1_value = tmp_v[idx1];
    int idx1_idx = tmp_i[idx1];
    tmp_v[idx1] = tmp_v[idx2];
    tmp_i[idx1] = tmp_i[idx2];
    tmp_v[idx2] = idx1_value;
    tmp_i[idx2] = idx1_idx;
}

__device__ void up(int cnt, float *__restrict__ tmp_v, int *__restrict__ tmp_i) {
    while (cnt) {
        if (tmp_v[cnt] > tmp_v[cnt >> 1]) {
            swap(cnt, cnt >> 1, tmp_v, tmp_i);
        } else {
            break;
        }
        cnt = cnt >> 1;
    }
}

__device__ void down(int cnt, const int heap_size, float *__restrict__ tmp_v, int *__restrict__ tmp_i) {
    while ((cnt << 1) < heap_size) {
        int left = cnt << 1, right = cnt << 1 | 1;
        int target = (right >= heap_size || tmp_v[left] > tmp_v[right]) ? left : right;

        if (tmp_v[cnt] <= tmp_v[target]) {
            swap(cnt, target, tmp_v, tmp_i);
        } else {
            break;
        }
        cnt = target;
    }
}

__device__ bool heap_insert(const float value, const int idx,
                            const int cnt, const int max_heap_size,
                            float *__restrict__ tmp_v,
                            int *__restrict__ tmp_i) {
    if (cnt < max_heap_size) { // not full
        tmp_v[cnt] = value;
        tmp_i[cnt] = idx;
        up(cnt, tmp_v, tmp_i);
        return true;
    }
    if (value >= *tmp_v) { // abandon if larger
        return false;
    }
    swap(0, cnt - 1, tmp_v, tmp_i); // delete heap top element
    down(0, cnt - 1, tmp_v, tmp_i); // down new top element
    tmp_v[cnt - 1] = value; // insert new element
    tmp_i[cnt - 1] = idx;
    up(cnt - 1, tmp_v, tmp_i);
    return false;
}