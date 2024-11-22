// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"
#include "crop.h"
#include "geometry.h"
#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_points", &gather_points);
    m.def("gather_points_grad", &gather_points_grad);
    m.def("furthest_point_sampling", &furthest_point_sampling);

    m.def("three_nn", &three_nn);
    m.def("three_interpolate", &three_interpolate);
    m.def("three_interpolate_grad", &three_interpolate_grad);

    m.def("ball_query", &ball_query);
    m.def("k_neighbor_query", &k_neighbor_query);
    m.def("graph_neighbor_query", &graph_neighbor_query);
    m.def("graph_neighbor_query_all", &graph_neighbor_query_all);

    m.def("group_points", &group_points);
    m.def("group_points_grad", &group_points_grad);

    m.def("random_crop", &random_crop);

    m.def("compute_vertex_normals", &compute_vertex_normals);
    m.def("compute_vertex_normals_grad", &compute_vertex_normals_grad);

    m.def("build_edge_matrix_from_triangle", &build_edge_matrix_from_triangle);
    m.def("build_graph_from_img", &build_graph_from_img);
}
