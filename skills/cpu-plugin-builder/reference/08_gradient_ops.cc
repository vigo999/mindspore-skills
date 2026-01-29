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
 * Template: Gradient Operations
 *
 * Category characteristics:
 * - Input includes grad_output (upstream gradient) and forward pass outputs/inputs
 * - Output is grad_input (gradient w.r.t. input)
 * - Uses ATen xxx_backward or xxx_backward_out functions
 *
 * Examples: sigmoid_grad, tanh_grad, relu_grad, softmax_grad, conv_grad
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Simple activation gradient (sigmoid, tanh, etc.)
 * Parameters: output (forward), grad_output, grad_input
 * Uses formula: grad_input = backward_fn(grad_output, output)
 */
extern "C" int {{OperatorName}}Grad(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                    void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto output = tensors[0];       /* output of forward pass */
  auto grad_output = tensors[1];  /* upstream gradient */
  auto grad_input = tensors[2];   /* gradient to compute */

  at::{{aten_function}}_backward_out(grad_input, grad_output, output);

  return 0;
}

/**
 * Variant B: Gradient using input instead of output (sqrt, etc.)
 * Parameters: input/output, grad_output, grad_input
 */
extern "C" int {{OperatorName}}GradFromOutput(int nparam, void **params, int *ndims, int64_t **shapes,
                                              const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self_or_output = tensors[0];
  auto grad_output = tensors[1];
  auto grad_input = tensors[nparam - 1];

  /* For sqrt: grad_input = grad_output / (2 * output) */
  at::{{aten_function}}_backward_out(grad_input, grad_output, self_or_output);

  return 0;
}

/**
 * Variant C: Complex gradient with multiple outputs (conv, pooling, etc.)
 * Parameters: dout, input, weight, bias (optional), ..., dx, dw, dbias
 */
extern "C" int {{OperatorName}}GradComplex(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                           void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto dout = tensors[0];
  auto input = tensors[1];
  auto weight = tensors[2];

  /* Handle optional bias */
  c10::optional<at::Tensor> bias = c10::nullopt;
  if (!input_utils.IsNoneInput(3)) {
    bias = tensors[3];
  }

  /* Get configuration parameters */
  auto stride = input_utils.GetIntVecInput(4);
  auto padding = input_utils.GetIntVecInput(5);

  /* Output gradients */
  auto dx = tensors[nparam - 3];
  auto dw = tensors[nparam - 2];
  auto dbias = tensors[nparam - 1];

  /* Call backward and copy results */
  auto result = at::{{aten_function}}_backward(dout, input, weight, /* ... */);

  dx.copy_(std::get<0>(result));
  dw.copy_(std::get<1>(result));
  if (bias.has_value()) {
    dbias.copy_(std::get<2>(result));
  }

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
