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
 * Template: Random Operations
 *
 * Category characteristics:
 * - Requires seed and offset for reproducibility
 * - Uses MakeCPUGenerator to create random generator
 * - May have shape parameter or input tensor
 *
 * Examples: randn, rand, randint, randperm, bernoulli, uniform, normal
 */

#include <torch/extension.h>

#include "utils/op_utils.h"
#include "utils/random_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Shape-based random generator (randn, rand)
 * Parameters: shape, seed, offset, output
 */
extern "C" int {{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto shape = input_utils.GetIntVecInput(0);
  auto at_output = tensors[nparam - 1];

  constexpr size_t seed_idx = 1;
  constexpr size_t offset_idx = 2;
  auto gen = MakeCPUGenerator(tensors[seed_idx], tensors[offset_idx]);

  (void)at::{{aten_function}}_out(at_output, shape, gen);

  return 0;
}

/**
 * Variant B: Tensor-based random (bernoulli, etc.)
 * Parameters: input (probabilities), seed, offset, output
 */
extern "C" int {{OperatorName}}FromTensor(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                          void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  constexpr size_t seed_idx = 1;
  constexpr size_t offset_idx = 2;
  auto gen = MakeCPUGenerator(tensors[seed_idx], tensors[offset_idx]);

  at::{{aten_function}}_out(at_output, at_input, gen);

  return 0;
}

/**
 * Variant C: Inplace random (uniform_, normal_)
 * Parameters: self, low, high, seed, offset
 */
extern "C" int Inplace{{OperatorName}}(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                       void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  double low = input_utils.GetScalarInput(1).toDouble();
  double high = input_utils.GetScalarInput(2).toDouble();

  constexpr size_t seed_idx = 3;
  constexpr size_t offset_idx = 4;
  auto gen = MakeCPUGenerator(tensors[seed_idx], tensors[offset_idx]);

  self.{{aten_function}}_(low, high, gen);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
