// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef POINTCONV_INTERPOLATE_H
#define POINTCONV_INTERPOLATE_H

#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight);
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m);

#endif //POINTCONV_DETACH_H
