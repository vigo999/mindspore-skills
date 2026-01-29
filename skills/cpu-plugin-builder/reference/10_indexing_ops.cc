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
 * Template: Indexing Operations
 *
 * Category characteristics:
 * - Involves index tensor(s) for selecting/scattering elements
 * - May have dimension parameter
 * - Operations include select, gather, scatter, index_select
 *
 * Examples: index_select, gather, scatter, scatter_add, index, narrow
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Index select
 * Parameters: input, dim, index, output
 */
extern "C" int IndexSelect(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_index = tensors[2];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(1);

  at::index_select_out(at_output, at_input, dim, at_index);

  return 0;
}

/**
 * Variant B: Gather
 * Parameters: input, dim, index, sparse_grad, output
 */
extern "C" int Gather(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_index = tensors[2];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(1);
  bool sparse_grad = input_utils.GetBoolInput(3);

  at::gather_out(at_output, at_input, dim, at_index, sparse_grad);

  return 0;
}

/**
 * Variant C: Inplace scatter (from src tensor)
 * Parameters: self, dim, index, src
 */
extern "C" int InplaceScatterSrc(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];
  auto index = tensors[2];
  auto src = tensors[3];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(1);

  self.scatter_(dim, index, src);

  return 0;
}

/**
 * Variant D: Inplace scatter add
 * Parameters: self, dim, index, src
 */
extern "C" int InplaceScatterAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto self = tensors[0];
  auto index = tensors[2];
  auto src = tensors[3];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t dim = input_utils.GetIntInput(1);

  self.scatter_add_(dim, index, src);

  return 0;
}

/**
 * Variant E: Narrow (view-based slicing)
 * Parameters: input, dim, start, length, output
 */
extern "C" int Narrow(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  int64_t dim = input_utils.GetIntInput(1);
  int64_t start = input_utils.GetIntInput(2);
  int64_t length = input_utils.GetIntInput(3);

  auto result = at::narrow(at_input, dim, start, length);
  at_output.copy_(result);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
