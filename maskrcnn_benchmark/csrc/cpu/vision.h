// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);


at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);


void modulated_deform_conv_cpu_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias);


void modulated_deform_conv_cpu_backward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor columns,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);


std::pair<at::Tensor, at::Tensor> soft_nms_cpu(const at::Tensor& dets,
                                               const at::Tensor& scores,
                                               const float threshold,
                                               const float sigma);
