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
 * Template: Linear Algebra Operations
 *
 * Category characteristics:
 * - Matrix operations (matmul, bmm, mm)
 * - Decompositions (svd, qr, eig, cholesky)
 * - Often have multiple outputs
 *
 * Examples: matmul, bmm, mm, dot, mv, addmm, addmv, svd, qr, det, norm
 */

#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {

/**
 * Variant A: Matrix multiplication (mm, bmm)
 * Parameters: input1, input2, output
 */
extern "C" int BatchMatMul(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input1 = tensors[0];
  auto at_input2 = tensors[1];
  auto at_output = tensors[nparam - 1];

  at::bmm_out(at_output, at_input1, at_input2);

  return 0;
}

/**
 * Variant B: Vector-Matrix operations (mv)
 * Parameters: mat, vec, output
 */
extern "C" int MatVec(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_mat = tensors[0];
  auto at_vec = tensors[1];
  auto at_output = tensors[nparam - 1];

  at::mv_out(at_output, at_mat, at_vec);

  return 0;
}

/**
 * Variant C: Addmm (beta*input + alpha*mat1@mat2)
 * Parameters: input, mat1, mat2, beta, alpha, output
 */
extern "C" int Addmm(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_mat1 = tensors[1];
  auto at_mat2 = tensors[2];
  auto at_output = tensors[nparam - 1];

  c10::Scalar beta = input_utils.GetScalarInput(3);
  c10::Scalar alpha = input_utils.GetScalarInput(4);

  at::addmm_out(at_output, at_input, at_mat1, at_mat2, beta, alpha);

  return 0;
}

/**
 * Variant D: Vector dot product
 * Parameters: input, other, output
 */
extern "C" int Dot(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_other = tensors[1];
  auto at_output = tensors[nparam - 1];

  auto result = at::dot(at_input, at_other);
  at_output.copy_(result);

  return 0;
}

/**
 * Variant E: SVD decomposition (multiple outputs)
 * Parameters: input, full_matrices, compute_uv, U, S, Vh
 */
extern "C" int SVD(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  auto at_input = tensors[0];
  auto at_U = tensors[nparam - 3];
  auto at_S = tensors[nparam - 2];
  auto at_Vh = tensors[nparam - 1];

  bool full_matrices = input_utils.GetBoolInput(1);

  at::linalg_svd_out(at_U, at_S, at_Vh, at_input, full_matrices);

  return 0;
}

/**
 * Variant F: Outer product
 * Parameters: input, vec2, output
 */
extern "C" int Outer(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];
  auto at_vec2 = tensors[1];
  auto at_output = tensors[nparam - 1];

  at::outer_out(at_output, at_input, at_vec2);

  return 0;
}

}  // namespace aten_op
}  // namespace op_plugin
