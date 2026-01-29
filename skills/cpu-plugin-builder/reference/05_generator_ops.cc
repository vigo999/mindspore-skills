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
 * Template: Generator Operations
 *
 * Category characteristics:
 * - No input tensor or only scalar parameters
 * - Creates new tensor from scratch
 * - Output shape specified via scalar parameters
 *
 * Examples: arange, linspace, zeros, ones, empty, eye, full
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Range-based generator (arange, linspace)
 * Parameters: start, end, step/steps, output
 */
extern "C" int {{OperatorName}}Range(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                     void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto start = input_utils.GetScalarInput(0);
  auto end = input_utils.GetScalarInput(1);
  int64_t steps = input_utils.GetIntInput(2);
  auto at_output = tensors[nparam - 1];

  at::{{aten_function}}_out(at_output, start, end, steps);

  return 0;
}

/**
 * Variant B: Shape-based generator (zeros, ones, empty)
 * Parameters: size/shape, output
 */
extern "C" int {{OperatorName}}Shape(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                     void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto size = input_utils.GetIntVecInput(0);
  auto at_output = tensors[nparam - 1];

  at::{{aten_function}}_out(at_output, size);

  return 0;
}

/**
 * Variant C: Eye matrix generator
 * Parameters: n, m (optional), output
 */
extern "C" int Eye(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t n = input_utils.GetIntInput(0);
  int64_t m = n;
  if (!input_utils.IsNoneInput(1)) {
    m = input_utils.GetIntInput(1);
  }
  auto at_output = tensors[nparam - 1];

  at::eye_out(at_output, n, m);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
