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
 * Template: Reduction Operations
 *
 * Category characteristics:
 * - Input tensor + dimension(s) parameter
 * - Optional keepdims (bool) parameter
 * - Optional dtype parameter
 * - Single tensor output (reduced shape)
 *
 * Examples: sum, mean, max, min, prod, amax, amin, all, any, norm
 */

#include <torch/extension.h>
#include <vector>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Reduce with single dimension
 */
extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  const auto &at_input = tensors[0];
  auto &at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t dim = input_utils.GetIntInput(1);
  bool keepdim = input_utils.GetBoolInput(2);

  at::{{aten_function}}_out(at_output, at_input, dim, keepdim);

  return 0;
}

/**
 * Variant B: Reduce with multiple dimensions (optional)
 */
extern "C" int {{OperatorName}}MultiDim(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                        void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  const auto &at_input = tensors[0];
  auto &at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  bool keepdim = input_utils.GetBoolInput(2);

  /* Handle optional dim parameter */
  if (input_utils.IsNoneInput(1)) {
    /* Reduce over all dimensions */
    at::{{aten_function}}_out(at_output, at_input, at::IntArrayRef{}, keepdim);
  } else {
    auto dim_vec = input_utils.GetIntVecInput(1);
    at::{{aten_function}}_out(at_output, at_input, at::IntArrayRef(dim_vec), keepdim);
  }

  return 0;
}

/**
 * Variant C: Reduce with dtype parameter
 */
extern "C" int {{OperatorName}}WithDtype(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                         void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  const auto &at_input = tensors[0];
  auto &at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  c10::optional<at::IntArrayRef> dim_opt = c10::nullopt;
  std::vector<int64_t> dim_vec;
  if (!input_utils.IsNoneInput(1)) {
    dim_vec = input_utils.GetIntVecInput(1);
    dim_opt = at::IntArrayRef(dim_vec);
  }

  bool keepdim = input_utils.GetBoolInput(2);
  c10::optional<at::ScalarType> dtype_opt = input_utils.GetOptionalScalarTypeInput(3);

  at::{{aten_function}}_out(at_output, at_input, dim_opt, keepdim, dtype_opt);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
