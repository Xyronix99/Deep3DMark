// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef POINTCONV_SAMPLING_H
#define POINTCONV_SAMPLING_H

#pragma once
#include <torch/extension.h>

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);

#endif