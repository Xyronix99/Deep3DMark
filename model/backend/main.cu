//
// Created by xingyu on 11/26/22.
//
#include "cuda_utils.h"
#include "crop.h"
#include "utils.h"
#include "ball_query.h"
#include "group_points.h"
#include "geometry.h"

#include <fstream>
#include <set>
#include <algorithm>
#include <vector>

namespace ti = torch::indexing;

std::vector<at::Tensor> read() {
    std::ifstream pc_file("/data_HDD/zhuxingyu/vscode/meshwatermark/pc.txt");
    std::ifstream face_file("/data_HDD/zhuxingyu/vscode/meshwatermark/faces.txt");
    std::ifstream normal_file("/data_HDD/zhuxingyu/vscode/meshwatermark/output/debug/normal.txt");
    defer {pc_file.close(); face_file.close(); normal_file.close(); };

    float a, b, c;
    int pc_cnt = 0, f1, f2, f3, f_cnt=0;
    std::vector<float> pc_vector;
    while (pc_file >> a >> b >> c) {
        ++pc_cnt;
        pc_vector.push_back(a);
        pc_vector.push_back(b);
        pc_vector.push_back(c);
    }
    std::vector<int> face_vector;
    while (face_file >> f1 >> f2 >> f3) {
        ++f_cnt;
        face_vector.push_back(f1);
        face_vector.push_back(f2);
        face_vector.push_back(f3);
    }

    std::vector<float> normal_vector;
    while (normal_file >> a >> b >> c) {
        normal_vector.push_back(a);
        normal_vector.push_back(b);
        normal_vector.push_back(c);
    }

    at::Tensor pc = torch::from_blob(pc_vector.data(), {1, pc_cnt, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor faces = torch::from_blob(face_vector.data(), {1, f_cnt, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();
    at::Tensor normal = torch::from_blob(normal_vector.data(), {1, pc_cnt, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    return std::vector<at::Tensor>{pc, faces, normal};
}

void test_crop_remain(at::Tensor remain_idx, at::Tensor cropped_idx) {
    int R = remain_idx.size(1);
    int S = cropped_idx.size(1);
    DEBUG("hi1");
    int * remain_idx_data = remain_idx.data_ptr<int>();
    DEBUG("hi2");
    int * cropped_idx_data = cropped_idx.data_ptr<int>();
    DEBUG("hi3");
    std::sort(remain_idx_data, remain_idx_data + R);
    DEBUG("hi4");
    std::sort(cropped_idx_data, cropped_idx_data + S);
    DEBUG("hi5");
    std::vector<int> out(std::max(R,S));
    auto inter_it = std::set_intersection(remain_idx_data, remain_idx_data + R, cropped_idx_data, cropped_idx_data + S, out.begin());
    auto union_it = std::set_union(remain_idx_data, remain_idx_data + R, cropped_idx_data, cropped_idx_data + S, out.begin());
    std::cout << "size_of_inter=" << inter_it - out.begin() << "size_of_union=" << union_it - out.begin() << std::endl;
}

void test_random_crop() {
    const int B = 4;
    const int N = 100000;
    const int M = 5000000;
    const int batch_idx = 3;

//    auto read_rtn = read();
//    at::Tensor pc = read_rtn[0];
//    at::Tensor faces = read_rtn[1];
    at::Tensor pc = torch::randn({B, N, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Float));
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Int));

    std::cout << "\n\n------Before cropping------" << std::endl;
//    std::cout << "points=" << pc[batch_idx] << std::endl;
//    std::cout << "faces=" << faces[batch_idx] << std::endl;

    auto rtn = random_crop(pc, faces, 0.3);
    auto cropped_pc = rtn[0].cpu();
    auto cropped_faces = rtn[1].cpu();
//    auto remain_idx = rtn[2].cpu();
//    auto cropped_idx = rtn[3].cpu();

    sleep(1);
    std::cout << "------After cropping------" << std::endl;

    std::cout << "points=" << cropped_pc[batch_idx] << std::endl;
    std::cout << "cropped_faces=" << cropped_faces[batch_idx].max() << std::endl;
//    std::cout << "remain_idx=" << remain_idx[batch_idx] << std::endl;
//    std::cout << "cropped_idx=" << cropped_idx[batch_idx] << std::endl;
}

void test_polygon_to_graph() {
    const int B = 1;
    const int N = 5;
    const int M = 10;
    const int batch_idx = 0;


    at::Tensor pc = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();

    auto rtn = build_graph_from_triangle(pc, faces);
    sleep(1);
    auto header_tensor = rtn[0][batch_idx].cpu();
    auto cnt_tensor = rtn[1][batch_idx].cpu();
    auto v_tensor = rtn[2][batch_idx].cpu();
    auto next_tensor = rtn[3][batch_idx].cpu();
    auto header = header_tensor.data_ptr<int>();
    auto cnt = cnt_tensor.data_ptr<int>();
    auto v = v_tensor.data_ptr<int>();
    auto next = next_tensor.data_ptr<int>();
    DEBUG(faces[batch_idx]);
    for (int p = 0; p < N; ++p) {
        std::cout << p << ":";
        for (int i=header[p]; ~i; i=next[i]) {
            std::cout << " " << v[i];
        }
        std::cout << std::endl;
    }
}


void test_graph_neighbor_query() {
    const int B = 4;
    const int N = 10;
    const int M = 8;
    const int batch_idx = 3;
    const int nsample = 9;


    at::Tensor pc = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();

    DEBUG(pc[batch_idx]);
    DEBUG(faces[batch_idx]);
    auto rtn = graph_neighbor_query(pc, faces, nsample);
    auto idx = rtn[0];
    auto num = rtn[1];
    sleep(1);
    DEBUG(idx[batch_idx]);
    DEBUG(num[batch_idx]);
//    auto new_pc = group_points(pc.permute({0, 2, 1}).contiguous(), idx).permute({0, 2, 3, 1});
//    DEBUG(new_pc[batch_idx]);
}


void test_compute_normals() {
    const int batch_idx = 0;
    auto rtn = read();
    auto pc = rtn[0];
    auto face = rtn[1];
    auto normal = rtn[2];
    auto rtn2 = compute_vertex_normals(pc, face);
    sleep(1);

    DEBUG(rtn2[0][batch_idx][242]);
    DEBUG(rtn2[1][batch_idx][242]);
}

void test_compute_vertex_normals_grad() {
    const int batch_idx = 0;
    auto rtn = read();
    torch::manual_seed(1);
    auto xyz = torch::randn({1, 5000, 3}).cuda();
    auto face = rtn[1];
    auto grad = torch::zeros({1, 5000, 3}).cuda();
    grad[0][0][0]=-0.4208;
    grad[0][0][1]=0.3361;
    grad[0][0][2]=0.0469;
    auto rtn1 = compute_vertex_normals(xyz, face);
    auto rtn2 = compute_vertex_normals_grad(grad, xyz, face, rtn1[1], rtn1[2]);
    sleep(1);
    DEBUG(rtn2[batch_idx][0]);
    DEBUG(rtn2[batch_idx].index({ti::Slice(0, 3)})); // (1, 3)
}

void build_graph() {
    const int B = 64;
    const int N = 788;
    const int M = 1577;
    const int batch_idx = 3;
    const int nsample = 9;


    at::Tensor pc = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();

//    DEBUG(pc[batch_idx]);
//    DEBUG(faces[batch_idx]);

    auto rtn = build_edge_matrix_from_triangle(pc, faces);
//    DEBUG(rtn[batch_idx]);
}

void build_graph_from_img() {
    const int B = 64;
    const int N = 10;
    const int batch_idx = 3;
    at::Tensor img = torch::randn({B, 3, N, N}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    auto rtn = build_graph_from_img(img);
    DEBUG(rtn[0][batch_idx]);
    DEBUG(rtn[1][batch_idx]);
}

void test_graph_neighbor_query_all() {
    const int B = 4;
    const int N = 10;
    const int M = 8;
    const int batch_idx = 3;


    at::Tensor pc = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();

    DEBUG(pc[batch_idx]);
    DEBUG(faces[batch_idx]);
    auto rtn = graph_neighbor_query_all(pc, faces);
    auto idx = rtn[0];
    auto num = rtn[1];
    sleep(1);
    DEBUG(idx[batch_idx]);
    DEBUG(num[batch_idx]);
}

int main(){
    test_graph_neighbor_query_all();
    return 0;
}