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
 * Template: Inplace Operations
 *
 * Category characteristics:
 * - Modifies input tensor directly (no separate output)
 * - Uses ATen xxx_() (underscore suffix) methods
 * - May have additional parameters
 *
 * Examples: copy_, fill_, add_, sub_, relu_, scatter_, index_put_
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Inplace copy
 * Parameters: self, src, non_blocking
 */
extern "C" int InplaceCopy(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];
  auto src = tensors[1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  bool non_blocking = input_utils.GetBoolInput(2);

  self.copy_(src, non_blocking);

  return 0;
}

/**
 * Variant B: Inplace fill with scalar
 * Parameters: self, value (scalar)
 */
extern "C" int InplaceFillScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  c10::Scalar value = input_utils.GetScalarInput(1);

  self.fill_(value);

  return 0;
}

/**
 * Variant C: Inplace fill with tensor
 * Parameters: self, value (tensor)
 */
extern "C" int InplaceFillTensor(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];
  auto value = tensors[1];

  self.fill_(value);

  return 0;
}

/**
 * Variant D: Inplace binary operation
 * Parameters: self, other, alpha (optional)
 */
extern "C" int Inplace{{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                       void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];
  auto other = tensors[1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  c10::Scalar alpha = 1;
  if (!input_utils.IsNoneInput(2)) {
    alpha = input_utils.GetScalarInput(2);
  }

  self.{{aten_function}}_(other, alpha);

  return 0;
}

/**
 * Variant E: Inplace activation
 * Parameters: self (modified inplace)
 */
extern "C" int Inplace{{OperatorName}}Activation(int nparam, void **params, int *ndims, int64_t **shapes,
                                                 const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];

  at::{{aten_function}}_(self);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
