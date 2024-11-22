//
// Created by xingyu on 11/29/22.
//

#ifndef POINTCONV_DETACH_H
#define POINTCONV_DETACH_H

#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> random_crop(at::Tensor, at::Tensor, const float);

#endif //POINTCONV_DETACH_H
