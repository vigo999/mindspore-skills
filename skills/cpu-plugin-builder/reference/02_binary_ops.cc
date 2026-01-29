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
 * Template: Binary Operations
 *
 * Category characteristics:
 * - Two tensor inputs
 * - Single tensor output
 * - Element-wise or broadcasting operation
 *
 * Examples: add, sub, mul, div, maximum, minimum, atan2, pow, logical_and, etc.
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_other = tensors[1];
  auto at_output = tensors[nparam - 1];

  at::{{aten_function}}_out(at_output, at_input, at_other);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
