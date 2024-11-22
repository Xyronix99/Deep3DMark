// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef POINTCONV_UTIL_H
#define POINTCONV_UTIL_H

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)


#define DEBUG(...) logger(#__VA_ARGS__, __VA_ARGS__)
template<typename ...Args>
void logger(std::string vars, Args&&... values) {
    std::cout << vars << " = ";
    std::string delim = "";
    (..., (std::cout << delim << values, delim = ", "));
    std::cout << std::endl;
}
//#define DEBUG(x) do { std::cout << #x << "=" << x; } while (0)

//////////////////////////////////////////////////////////////////
////////////                polygon to graph                /////////////
//////////////////////////////////////////////////////////////////

std::vector<at::Tensor> build_graph_from_triangle(at::Tensor xyz, at::Tensor polygons);

at::Tensor build_edge_matrix_from_triangle(at::Tensor xyz, at::Tensor faces);

std::vector<at::Tensor> build_graph_from_img(at::Tensor img);

#endif



#ifndef defer
struct defer_dummy {};
template <class F> struct deferrer { F f; ~deferrer() { f(); } };
template <class F> deferrer<F> operator*(defer_dummy, F f) { return {f}; }
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} *[&]()
#endif // defer