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
 * Template: Comparison Operations
 *
 * Category characteristics:
 * - Compares two tensors element-wise
 * - Output is boolean tensor
 * - Similar to binary ops but output dtype is bool
 *
 * Examples: eq, ne, lt, le, gt, ge, equal, isclose
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Tensor-Tensor comparison
 * Parameters: input, other, output (bool tensor)
 */
extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_other = tensors[1];
  auto at_output = tensors[nparam - 1];

  at::{{aten_function}}_out(at_output, at_input, at_other);

  return 0;
}

/**
 * Variant B: Tensor-Scalar comparison
 * Parameters: input, scalar, output (bool tensor)
 */
extern "C" int {{OperatorName}}Scalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                      void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  c10::Scalar other = input_utils.GetScalarInput(1);

  at::{{aten_function}}_out(at_output, at_input, other);

  return 0;
}

/**
 * Variant C: IsClose (with tolerance parameters)
 * Parameters: input, other, rtol, atol, equal_nan, output
 */
extern "C" int IsClose(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                       void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_other = tensors[1];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  double rtol = input_utils.GetScalarInput(2).toDouble();
  double atol = input_utils.GetScalarInput(3).toDouble();
  bool equal_nan = input_utils.GetBoolInput(4);

  auto result = at::isclose(at_input, at_other, rtol, atol, equal_nan);
  at_output.copy_(result);

  return 0;
}

/**
 * Variant D: Tensor equality check (returns single bool)
 * Parameters: input, other, output (0-dim bool tensor)
 */
extern "C" int Equal(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_other = tensors[1];
  auto at_output = tensors[nparam - 1];

  bool result = at::equal(at_input, at_other);
  at_output.fill_(result);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
