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
 * Template: Shape Operations
 *
 * Category characteristics:
 * - Manipulates tensor shape/layout
 * - Includes concatenation, stacking, splitting, reshaping
 * - May involve multiple input tensors or dimension parameters
 *
 * Examples: concat, stack, split, chunk, tile, repeat, flatten, reshape
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Concatenation
 * Parameters: tensor1, tensor2, ..., dim, output
 * Note: dim is the second-to-last parameter
 */
extern "C" int Concat(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_output = tensors[nparam - 1];

  /* Remove output and dim from tensors list, keep only inputs */
  tensors.resize(nparam - 2);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(nparam - 2);

  at::cat_out(at_output, tensors, dim);

  return 0;
}

/**
 * Variant B: Stack
 * Parameters: tensor1, tensor2, ..., dim, output
 */
extern "C" int Stack(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_output = tensors[nparam - 1];
  tensors.resize(nparam - 2);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(nparam - 2);

  at::stack_out(at_output, tensors, dim);

  return 0;
}

/**
 * Variant C: Tile/Repeat
 * Parameters: input, reps, output
 */
extern "C" int Tile(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                    void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  auto reps = input_utils.GetIntVecInput(1);

  auto result = at::tile(at_input, reps);
  at_output.copy_(result);

  return 0;
}

/**
 * Variant D: Flatten
 * Parameters: input, start_dim, end_dim, output
 */
extern "C" int Flatten(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                       void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t start_dim = input_utils.GetIntInput(1);
  int64_t end_dim = input_utils.GetIntInput(2);

  auto result = at::flatten(at_input, start_dim, end_dim);
  at_output.copy_(result);

  return 0;
}

/**
 * Variant E: Transpose/Permute
 * Parameters: input, dim0, dim1, output (for transpose)
 * OR: input, dims, output (for permute)
 */
extern "C" int Transpose(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t dim0 = input_utils.GetIntInput(1);
  int64_t dim1 = input_utils.GetIntInput(2);

  auto result = at::transpose(at_input, dim0, dim1);
  at_output.copy_(result);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
