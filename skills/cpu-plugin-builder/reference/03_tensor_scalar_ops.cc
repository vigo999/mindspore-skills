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
 * Template: Tensor-Scalar Operations
 *
 * Category characteristics:
 * - One tensor input + one or more scalar parameters
 * - Single tensor output
 * - Scalar may be passed as 0-dim tensor or via KernelInputUtils
 *
 * Examples: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_tensor_scalar
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Scalar as 0-dim tensor
 * Use when scalar is passed as a tensor in parameter list
 */
extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  /* Extract scalar from 0-dim tensor */
  c10::Scalar scalar_value = tensors[1].item();

  at::{{aten_function}}_out(at_output, at_input, scalar_value);

  return 0;
}

/**
 * Variant B: Scalar via KernelInputUtils
 * Use when scalar is passed via extra info
 */
extern "C" int {{OperatorName}}WithUtils(int nparam, void **params, int *ndims, int64_t **shapes,
                                         const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  /* Extract scalar using input_utils */
  c10::Scalar scalar_value = input_utils.GetScalarInput(1);

  at::{{aten_function}}_out(at_output, at_input, scalar_value);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
