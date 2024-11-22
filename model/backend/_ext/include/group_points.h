// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef POINTCONV_GROIP_POINTS_H
#define POINTCONV_GROIP_POINTS_H

#pragma once
#include <torch/extension.h>

at::Tensor group_points(at::Tensor points, at::Tensor idx);
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

#endif