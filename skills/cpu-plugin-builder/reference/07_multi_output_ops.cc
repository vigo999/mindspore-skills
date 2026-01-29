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
 * Template: Multi-output Operations
 *
 * Category characteristics:
 * - Multiple output tensors (e.g., values + indices)
 * - Outputs are usually at the end of parameter list
 * - ATen function takes multiple output references
 *
 * Examples: max (with dim), min (with dim), cummax, cummin, topk, sort, svd, qr
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Two outputs (values, indices) - common for max/min with dim
 * Parameters: input, dim, keepdim, values, indices
 */
extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_values = tensors[nparam - 2];
  auto at_indices = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t dim = input_utils.GetIntInput(1);
  bool keepdim = input_utils.GetBoolInput(2);

  at::{{aten_function}}_out(at_values, at_indices, at_input, dim, keepdim);

  return 0;
}

/**
 * Variant B: Cumulative operations (cummax, cummin)
 * Parameters: input, dim, values, indices
 */
extern "C" int {{OperatorName}}Cumulative(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                          void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_values = tensors[nparam - 2];
  auto at_indices = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t dim = input_utils.GetIntInput(1);

  at::{{aten_function}}_out(at_values, at_indices, at_input, dim);

  return 0;
}

/**
 * Variant C: TopK operation
 * Parameters: input, k, dim, largest, sorted, values, indices
 */
extern "C" int TopK(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                    void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_values = tensors[nparam - 2];
  auto at_indices = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t k = input_utils.GetIntInput(1);
  int64_t dim = input_utils.GetIntInput(2);
  bool largest = input_utils.GetBoolInput(3);
  bool sorted = input_utils.GetBoolInput(4);

  at::topk_out(at_values, at_indices, at_input, k, dim, largest, sorted);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
