//
// Created by xingyu on 12/14/22.
//

#include "cuda_utils.h"


__device__ bool check_exist(int from, int to,
                            int *__restrict__ header,
                            int *__restrict__ cnt,
                            int *__restrict__ v,
                            int *__restrict__ next) {
    for (int i = header[from]; i != -1; i = next[i]) {
        if (v[i] == to) {
            return true;
        }
    }
    return false;
}

__device__ void add_edge(int from, int to,
                         int *__restrict__ header,
                         int *__restrict__ cnt,
                         int *__restrict__ v,
                         int *__restrict__ next) {
    if (check_exist(from, to, header, cnt, v, next)) {
        return;
    }
    int c = cnt[0]++;
    v[c] = to;
    next[c] = header[from];
    header[from] = c;
}
/**
 *
 * @param B batch size
 * @param N number of vertices
 * @param M number of polygons
 * @param polygons (B, M, 3)
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, 6M)
 * @param next (B, 6M)
 */
__global__ void build_graph_from_triangle_kernel(int B, int N, int M, const int *__restrict__ polygons,
                                   int *__restrict__ header,
                                   int *__restrict__ cnt,
                                   int *__restrict__ v,
                                   int *__restrict__ next) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        // has to be sequential execution
        for (int m = 0; m < M; ++m) {
            int x = polygons[(b * M + m) * 3], y = polygons[(b * M + m) * 3 + 1], z = polygons[(b * M + m) * 3 + 2];
            if (x == -1) {
                break;
            }
            add_edge(x, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(y, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(x, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(z, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(y, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
            add_edge(z, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M);
        }
    }
}

/**
 *
 * @param B batch size
 * @param N number of vertices
 * @param M number of polygons
 * @param polygons (B, M, 3)
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, 6M)
 * @param next (B, 6M)
 */
void build_graph_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ polygons,
                                int *__restrict__ header,
                                int *__restrict__ cnt,
                                int *__restrict__ v,
                                int *__restrict__ next) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_graph_from_triangle_kernel<<<B, 1, 0, stream>>>(B, N, M, polygons, header, cnt, v, next);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param faces (B, M, 3)
 * @param out (B, N, N)
 */
__global__ void build_edge_matrix_from_triangle_kernel(int B, int N, int M, const int *__restrict__ faces, int *__restrict__ out) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            int x = faces[(b * M + m) * 3], y = faces[(b * M + m) * 3 + 1], z = faces[(b * M + m) * 3 + 2];
            if (x==-1 || y==-1 || z==-1) {
                continue;
            }
            out[(b * N + x) * N + y] = out[(b * N + y) * N + x] \
            = out[(b * N + x) * N + z] = out[(b * N + z) * N + x] \
            = out[(b * N + z) * N + y] = out[(b * N + y) * N + z] = 1;
        }
    }

    __syncthreads();

    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int m = threadIdx.x; m < N; m += blockDim.x) {
            out[(b * N + m) * N + m] = 0;
        }
    }
}

void build_edge_matrix_from_triangle_kernel_wrapper(int B, int N, int M, const int *__restrict__ faces, int *__restrict__ out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_edge_matrix_from_triangle_kernel<<<B, opt_n_threads(M), 0, stream>>>(B, N, M, faces, out);

    CUDA_CHECK_ERRORS();
}

/**
 *
 * @param B
 * @param N
 * @param M
 * @param img (B, 3, H, W)
 * @param xyz (B, N, 3)
 * @param idx (B, N, 9)
 */
__global__ void build_graph_from_img_kernel(int B, int H, int W, const float *__restrict__ img, float *__restrict__ xyz, int *__restrict__ idx) {
    int N = H * W;
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            int h = n / W, w = n % W;
            for (int i = 0; i < 3; ++i) {
                xyz[(b * N + n) * 3 + i] = img[b * 3 * H * W + i * H * W + h * W + w];
            }
            int cnt = 0;
            for (int i1 = -1; i1 <= 1; i1++) {
                for (int i2 = -1; i2 <=1 ;i2++) {
                    int hh = h+i1, ww = w+i2;
                    if (hh<0 || ww<0 || hh>=H || ww>=W) continue;
                    idx[(b * N + n) * 9 + cnt++] = hh * W + ww;
                }
            }
        }
    }
}

void build_graph_from_img_kernel_wrapper(int B, int H, int W, const float *__restrict__ img, float *__restrict__ xyz, int *__restrict__ idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_graph_from_img_kernel<<<B, opt_n_threads(H * W), 0, stream>>>(B, H, W, img, xyz, idx);

    CUDA_CHECK_ERRORS();
}