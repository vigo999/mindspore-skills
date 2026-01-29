/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Template: Neural Network Operations
 *
 * Category characteristics:
 * - Complex operations with multiple tensor inputs (input, weight, bias)
 * - Multiple configuration parameters (stride, padding, dilation, groups)
 * - Often have optional parameters
 *
 * Examples: conv2d, conv3d, linear, batch_norm, layer_norm, pooling, softmax
 */

#include <torch/extension.h>
#include <vector>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Convolution (2D/3D)
 * Parameters: input, weight, bias (optional), stride, padding, dilation, groups, output
 */
extern "C" int Conv2D(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_weight = tensors[1];
  auto at_output = tensors[nparam - 1];

  /* Handle optional bias */
  c10::optional<at::Tensor> at_bias = c10::nullopt;
  if (!input_utils.IsNoneInput(2)) {
    at_bias = tensors[2];
  }

  auto stride = input_utils.GetIntVecInput(3);
  auto padding = input_utils.GetIntVecInput(4);
  auto dilation = input_utils.GetIntVecInput(5);
  int64_t groups = input_utils.GetIntInput(6);

  /* For 2D conv: output_padding = {0, 0}, transposed = false */
  std::vector<int64_t> output_padding = {0, 0};
  at::convolution_out(at_output, at_input, at_weight, at_bias, stride, padding, dilation, false, output_padding,
                      groups);

  return 0;
}

/**
 * Variant B: Pooling operations (avg_pool, max_pool)
 * Parameters: input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, output
 */
extern "C" int AvgPool2D(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  auto kernel_size = input_utils.GetIntVecInput(1);

  at::IntArrayRef stride;
  if (!input_utils.IsNoneInput(2)) {
    stride = input_utils.GetIntVecInput(2);
  } else {
    stride = kernel_size;
  }

  at::IntArrayRef padding;
  if (!input_utils.IsNoneInput(3)) {
    padding = input_utils.GetIntVecInput(3);
  } else {
    padding = {0, 0};
  }

  bool ceil_mode = input_utils.GetBoolInput(4);
  bool count_include_pad = input_utils.GetBoolInput(5);

  c10::optional<int64_t> divisor_override = c10::nullopt;
  if (!input_utils.IsNoneInput(6)) {
    divisor_override = input_utils.GetScalarInput(6).toLong();
  }

  at::avg_pool2d_out(at_output, at_input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);

  return 0;
}

/**
 * Variant C: Softmax operation
 * Parameters: input, dim, dtype (optional), output
 */
extern "C" int Softmax(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                       void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  int64_t dim = input_utils.GetIntInput(1);

  c10::optional<at::ScalarType> dtype = c10::nullopt;
  if (!input_utils.IsNoneInput(2)) {
    dtype = input_utils.GetOptionalScalarTypeInput(2);
  }

  at::_softmax_out(at_output, at_input, dim, false);

  return 0;
}

/**
 * Variant D: Batch/Layer Normalization
 * Parameters: input, weight, bias, running_mean, running_var, training, momentum, eps, output
 */
extern "C" int BatchNorm(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  c10::optional<at::Tensor> weight = c10::nullopt;
  c10::optional<at::Tensor> bias = c10::nullopt;
  c10::optional<at::Tensor> running_mean = c10::nullopt;
  c10::optional<at::Tensor> running_var = c10::nullopt;

  if (!input_utils.IsNoneInput(1)) {
    weight = tensors[1];
  }
  if (!input_utils.IsNoneInput(2)) {
    bias = tensors[2];
  }
  if (!input_utils.IsNoneInput(3)) {
    running_mean = tensors[3];
  }
  if (!input_utils.IsNoneInput(4)) {
    running_var = tensors[4];
  }

  bool training = input_utils.GetBoolInput(5);
  double momentum = input_utils.GetScalarInput(6).toDouble();
  double eps = input_utils.GetScalarInput(7).toDouble();

  auto result = at::batch_norm(at_input, weight, bias, running_mean, running_var, training, momentum, eps, true);
  at_output.copy_(result);

  return 0;
}

/**
 * Variant E: Activation functions with parameters (leaky_relu, elu, hardtanh)
 * Parameters: input, negative_slope/alpha, output
 */
extern "C" int LeakyReLU(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  double negative_slope = input_utils.GetScalarInput(1).toDouble();

  at::leaky_relu_out(at_output, at_input, negative_slope);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
