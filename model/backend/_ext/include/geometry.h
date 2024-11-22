//
// Created by xingyu on 1/10/23.
//

#ifndef POINTCONV_GEOMETRY_H
#define POINTCONV_GEOMETRY_H

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> compute_vertex_normals(at::Tensor xyz, at::Tensor faces);
at::Tensor compute_vertex_normals_grad(at::Tensor vertex_normal_grad, at::Tensor xyz, at::Tensor faces, at::Tensor normals, at::Tensor norm);

#endif //POINTCONV_GEOMETRY_H
