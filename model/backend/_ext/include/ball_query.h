// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>
#include <vector>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);

at::Tensor k_neighbor_query(at::Tensor new_xyz, at::Tensor xyz, const int nsample);

std::vector<at::Tensor> graph_neighbor_query(at::Tensor xyz, at::Tensor faces, const int nsample);


std::vector<at::Tensor> graph_neighbor_query_all(at::Tensor xyz, at::Tensor faces);