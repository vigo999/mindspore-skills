# mindspore_op_plugin development process

## 1. Create Issues and Pull Requests

- Create a branch from `master` named after the operator. Examples:
  - `git checkout -b linspace_op`
  - `git checkout -b sub_ext_op`

- On Gitee, pick the Feature template. Issue title format: `[Feature] [OP Plugin] [OPS] [CPU] <op name> <brief>`. Examples:
  - `[Feature] [OP Plugin] [OPS] [CPU] mint.linspace op enablement`
  - `[Feature] [OP Plugin] [OPS] [CPU] Tensor.sub_ op enablement`

- Add an Issue description. Examples:
  - [isses_linspace_op.md](./isses_linspace_op.md)
  - [issuse_sub_inplace_op.md](./issuse_sub_inplace_op.md)

- In the Issue’s `Design Idea` section, state the target ATen operator/overload and the core adaptation path.

- Create Pull Requests and link the corresponding Issue.

## 2. Basic Lookups

- **Operator name lookup**: In the `mindspore` repo under `mindspore/ops/op_def/yaml`, find the definition and derive the operator function name by converting the YAML function (snake_case) to CamelCase. Examples:
  - In `mindspore/ops/op_def/yaml/lin_space_ext_op.yaml`, if the function is `lin_space_ext` in yaml, the operator name should be `LinSpaceExt`.
  - In `mindspore/ops/op_def/yaml/sub_ext_op.yaml`, if the function is `sub_ext` in yaml, the operator name should be `SubExt`.

- **ATen operator lookup**: Search `aten/src/ATen/native/native_functions.yaml` in the `pytorch` repo.

- **Align with ATen**: Use the overload list in `aten/src/ATen/native/native_functions.yaml`; for `_out` variants, refer to `aten/src/ATen/templates/RedispatchFunctions.h`. Example: 
  - search `cat` in `native_functions.yaml` and use the `at::cat_out` variant.

## 3. Operator Implementation

> Notes: Development happens mainly in the `mindspore_op_plugin` repo. Operators live under `op_plugin/ops/kernel/*.cc`, with paired tests in `tests/st/mint/test_*.py` and `tests/st/mint/test_perf_*.py`. CMake (`cmake/generate_op_registry.cmake`) scans `extern "C" int Xxx(...)` in `op_plugin/ops/kernel/*.cc`, generates registry headers, and integrates into MindSpore.

> How it works: `ConvertToATenTensors(..., c10::kCPU)` converts `mindspore.Tensor` to `torch.Tensor`. `KernelInputInfo` + `KernelInputUtils` parse non-Tensor parameters. Convention: `tensors[nparam - 1]` is the output. Then call the ATen `_out` implementation.

- Add `*.cc` file under `op_plugin/ops/kernel/` following the operator name, for example:
  - `op_plugin/ops/kernel/linspace_ext.cc`
  - `op_plugin/ops/kernel/inplace_sub_ext.cc`
  - `op_plugin/ops/kernel/inplace_sub_scalar.cc`

- Implement the core operator code. Examples:

  - `op_plugin/ops/kernel/linspace_ext.cc`

    ```c++
    /**
    * Copyright 2025 Huawei Technologies Co., Ltd
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
    #include <string.h>
    #include <torch/extension.h>
    #include <iostream>

    #include "utils/op_utils.h"

    namespace op_plugin {
    namespace aten_op {
    extern "C" int LinSpaceExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                              void *extra) {
      auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

      KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
      KernelInputUtils input_utils(input_info);

      constexpr size_t start_idx = 0;
      constexpr size_t end_idx = 1;
      constexpr size_t steps_idx = 2;

      at::Scalar start = input_utils.GetScalarInput(start_idx);
      at::Scalar end = input_utils.GetScalarInput(end_idx);
      int64_t steps = input_utils.GetIntInput(steps_idx);

      auto output = tensors[nparam - 1];

      at::linspace_out(output, start, end, steps);
      return 0;
    }
    }  // namespace aten_op
    }  // namespace op_plugin
    ```

  - `op_plugin/ops/kernel/inplace_sub_ext.cc`

    ```c++
    /**
    * Copyright 2025 Huawei Technologies Co., Ltd
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
    #include <string.h>
    #include <torch/extension.h>
    #include <iostream>

    #include "utils/op_utils.h"

    namespace op_plugin {
    namespace aten_op {
    extern "C" int InplaceSubExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
      // Parameter list: [self, other (tensor), alpha (scalar), output]
      auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
      auto self = tensors[0];
      auto other = tensors[1];

      // Extract alpha parameter (non-Tensor)
      KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
      KernelInputUtils input_utils(input_info);
      c10::Scalar alpha_scalar = input_utils.GetScalarInput(2);

      // Call ATen inplace interface: self -= alpha * other
      self.sub_(other, alpha_scalar);
      return 0;
    }
    }  // namespace aten_op
    }  // namespace op_plugin
    ```

  - `op_plugin/ops/kernel/inplace_sub_scalar.cc`

    ```c++
    /**
    * Copyright 2025 Huawei Technologies Co., Ltd
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
    #include <string.h>
    #include <torch/extension.h>
    #include <iostream>

    #include "utils/op_utils.h"

    namespace op_plugin {
    namespace aten_op {
    extern "C" int InplaceSubScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                    void *stream, void *extra) {
      // Parameter list: [self, other (scalar), alpha (scalar), output]
      auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
      auto self = tensors[0];

      // Extract scalar parameters
      KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
      KernelInputUtils input_utils(input_info);
      auto scalar_other = input_utils.GetScalarInput(1);
      auto scalar_alpha = input_utils.GetScalarInput(2);

      // Call ATen inplace interface: self -= alpha * other
      self.sub_(scalar_other, scalar_alpha);
      return 0;
    }
    }  // namespace aten_op
    }  // namespace op_plugin
    ```

- Coding style: no tabs or trailing spaces; line length ≤ 120; add necessary comments.

## 4. Build and Artifacts

- Auto-registration is already integrated in `CMakeLists.txt`.
- Build:
  - Linux: `source env.source && bash build.sh`; for example, run on Ubuntu and check the log for `Found operator: LinSpaceExt` to confirm auto-registration.
  - Windows: run `env.bat`, then `build.bat`.
- Artifact: `build/ms_op_plugin.(so/dll)`. Before running, set `MS_OP_PLUGIN_PATH` to the artifact. Example: `export MS_OP_PLUGIN_PATH=/path/to/build/ms_op_plugin.so`.

## 5. Verification

- Functional tests: benchmark against torch; prefer ATen `*_out`; use `allclose_nparray(..., equal_nan=True)` for numerical checks. Add `tests/st/mint/test_*.py` under `tests/st/mint/`. Example for linspace: add `tests/st/mint/test_linspace.py` and run `source env.source && pytest tests/st/mint/test_linspace.py`:

  ```python
  #!/usr/bin/env python3
  # Copyright 2025 Huawei Technologies Co., Ltd
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing,
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.
  # ============================================================================
  """ linspace op test case """
  import pytest
  import numpy as np
  import mindspore as ms
  from mindspore import mint, jit
  from tests.utils.mark_utils import arg_mark
  from tests.utils.tools import allclose_nparray
  import torch


  def generate_expect_forward_output(start, end, steps, dtype=None):
      """Generate expected output using PyTorch linspace."""
      if dtype is None:
          return torch.linspace(start, end, steps)
      if dtype == ms.float16:
          torch_dtype = torch.float16
      elif dtype == ms.float32:
          torch_dtype = torch.float32
      elif dtype == ms.float64:
          torch_dtype = torch.float64
      else:
          torch_dtype = None
      return torch.linspace(start, end, steps, dtype=torch_dtype)


  def linspace_forward_func(start, end, steps, dtype=None):
      """Forward function for mint.linspace."""
      if dtype is None:
          return mint.linspace(start, end, steps)
      return mint.linspace(start, end, steps, dtype=dtype)


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level0",
      card_mark="onecard",
      essential_mark="essential",
  )
  @pytest.mark.parametrize("mode", ["pynative", "KBK"])
  @pytest.mark.parametrize("dtype", [None, ms.float32])
  def test_linspace_std(mode, dtype):
      """
      Feature: pyboost function.
      Description: test function linspace.
      Expectation: expect correct result.
      """
      np.random.seed(0)
      start = float(np.random.uniform(-10.0, 0.0))
      end = float(np.random.uniform(0.1, 10.0))
      steps = int(np.random.randint(2, 100))

      expect = generate_expect_forward_output(start, end, steps, dtype)

      if mode == "pynative":
          ms.context.set_context(mode=ms.PYNATIVE_MODE)
          output = linspace_forward_func(start, end, steps, dtype)
      elif mode == "KBK":
          output = jit(
              linspace_forward_func,
              backend="ms_backend",
              jit_level="O0",
          )(start, end, steps, dtype)
      else:
          output = None

      allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level0",
      card_mark="onecard",
      essential_mark="essential",
  )
  @pytest.mark.parametrize("mode", ["pynative", "KBK"])
  @pytest.mark.parametrize("dtype", [ms.float16, ms.float32, ms.float64])
  def test_linspace_dtype_coverage(mode, dtype):
      """
      Feature: dtype coverage for linspace operator.
      Description: test linspace with various floating dtypes.
      Expectation: results match PyTorch implementation.
      """
      np.random.seed(1)
      start = float(np.random.uniform(-100.0, -1.0))
      end = float(np.random.uniform(1.0, 100.0))
      steps = int(np.random.randint(2, 50))

      expect = generate_expect_forward_output(start, end, steps, dtype)

      if mode == "pynative":
          ms.context.set_context(mode=ms.PYNATIVE_MODE)
          output = linspace_forward_func(start, end, steps, dtype)
      elif mode == "KBK":
          output = jit(
              linspace_forward_func,
              backend="ms_backend",
              jit_level="O0",
          )(start, end, steps, dtype)
      else:
          output = None

      allclose_nparray(
          expect.detach().numpy(),
          output.asnumpy(),
          rtol=2e-6,
          atol=2e-6,
          equal_nan=True,
      )


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level0",
      card_mark="onecard",
      essential_mark="essential",
  )
  @pytest.mark.parametrize("mode", ["pynative", "KBK"])
  def test_linspace_reverse_interval(mode):
      """
      Feature: reverse interval support for linspace.
      Description: test linspace when start is greater than end.
      Expectation: results match PyTorch implementation.
      """
      np.random.seed(2)
      start = float(np.random.uniform(1.0, 10.0))
      end = float(np.random.uniform(-10.0, -1.0))
      steps = int(np.random.randint(2, 100))

      expect = generate_expect_forward_output(start, end, steps, ms.float32)

      if mode == "pynative":
          ms.context.set_context(mode=ms.PYNATIVE_MODE)
          output = linspace_forward_func(start, end, steps, ms.float32)
      elif mode == "KBK":
          output = jit(
              linspace_forward_func,
              backend="ms_backend",
              jit_level="O0",
          )(start, end, steps, ms.float32)
      else:
          output = None

      allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level0",
      card_mark="onecard",
      essential_mark="essential",
  )
  @pytest.mark.parametrize("mode", ["pynative", "KBK"])
  @pytest.mark.parametrize(
      "start_end_pair",
      [
          (0.0, 1.0),
          (-1.0, 0.0),
          (-5.0, 5.0),
          (-1e3, 1e3),
          (-1e-5, 1e-5),
      ],
  )
  def test_linspace_numeric_ranges(mode, start_end_pair):
      """
      Feature: numeric range coverage for linspace.
      Description: test linspace with different numeric ranges.
      Expectation: results match PyTorch implementation.
      """
      start, end = start_end_pair
      steps = 50

      expect = generate_expect_forward_output(start, end, steps, ms.float32)

      if mode == "pynative":
          ms.context.set_context(mode=ms.PYNATIVE_MODE)
          output = linspace_forward_func(start, end, steps, ms.float32)
      elif mode == "KBK":
          output = jit(
              linspace_forward_func,
              backend="ms_backend",
              jit_level="O0",
          )(start, end, steps, ms.float32)
      else:
          output = None

      # Check endpoints and overall values
      out_np = output.asnumpy()
      expect_np = expect.detach().numpy()
      allclose_nparray(expect_np, out_np, equal_nan=True)
      assert out_np.shape[0] == steps
      assert np.allclose(out_np[0], start)
      assert np.allclose(out_np[-1], end)


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level0",
      card_mark="onecard",
      essential_mark="essential",
  )
  @pytest.mark.parametrize("mode", ["pynative", "KBK"])
  @pytest.mark.parametrize(
      "start, end",
      [
          (float("nan"), 1.0),
          (0.0, float("nan")),
          (float("inf"), 1.0),
          (-float("inf"), -1.0),
      ],
  )
  def test_linspace_special_values(mode, start, end):
      """
      Feature: special value handling for linspace.
      Description: test linspace with nan and inf endpoints.
      Expectation: results match PyTorch implementation.
      """
      steps = 10

      expect = generate_expect_forward_output(start, end, steps, ms.float32)

      if mode == "pynative":
          ms.context.set_context(mode=ms.PYNATIVE_MODE)
          output = linspace_forward_func(start, end, steps, ms.float32)
      elif mode == "KBK":
          output = jit(
              linspace_forward_func,
              backend="ms_backend",
              jit_level="O0",
          )(start, end, steps, ms.float32)
      else:
          output = None

      allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
  ```

- Performance tests: benchmark against torch; measure end-to-end time over multiple runs. After subtracting `BACKGROUND_NOISE`, MindSpore CPU E2E time should be ≤ 1.1x torch. Add `_pynative_executor.sync()` before/after MindSpore timing. Add `tests/st/mint/test_perf_*.py`. Example for linspace: add `tests/st/mint/test_perf_linspace.py` and run `source env.source && pytest tests/st/mint/test_perf_linspace.py`:

  ```python
  #!/usr/bin/env python3
  # Copyright 2025 Huawei Technologies Co., Ltd
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing,
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.
  # ============================================================================
  """ linspace op performance test case """
  # pylint: disable=unused-variable
  # pylint: disable=W0622,W0613
  import time
  import numpy as np
  import mindspore as ms
  from mindspore import mint
  from mindspore.common.api import _pynative_executor
  from tests.utils.test_op_utils import BACKGROUND_NOISE
  from tests.utils.mark_utils import arg_mark
  import torch
  import pytest


  def linspace_forward_perf(start, end, steps):
      """Get MindSpore linspace forward performance."""
      # Warm-up
      for _ in range(1000):
          _ = mint.linspace(start, end, steps)

      _pynative_executor.sync()
      start_time = time.time()
      # Performance test
      for _ in range(1000):
          _ = mint.linspace(start, end, steps)
      _pynative_executor.sync()
      end_time = time.time()

      return end_time - start_time


  def generate_expect_linspace_forward_perf(start, end, steps):
      """Get PyTorch linspace forward performance."""
      # Warm-up
      for _ in range(1000):
          _ = torch.linspace(start, end, steps)

      start_time = time.time()
      # Performance test
      for _ in range(1000):
          _ = torch.linspace(start, end, steps)
      end_time = time.time()

      return end_time - start_time


  @arg_mark(
      plat_marks=["cpu_linux"],
      level_mark="level1",
      card_mark="onecard",
      essential_mark="unessential",
  )
  @pytest.mark.parametrize("mode", ["pynative"])
  def test_linspace_perf(mode):
      """
      Feature: standard forward performance for linspace.
      Description: test linspace op performance.
      Expectation: expect performance OK.
      """
      del mode

      ms.context.set_context(mode=ms.PYNATIVE_MODE)

      # Generate random start and end for performance test
      np.random.seed(0)
      start = float(np.random.uniform(-10.0, 0.0))
      end = float(np.random.uniform(0.1, 10.0))
      steps = int(np.random.randint(100000, 200000))

      ms_perf = linspace_forward_perf(start, end, steps)
      expect_perf = generate_expect_linspace_forward_perf(start, end, steps)
      assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
  ```

- Test coverage expectations: default arguments; empty tensors; NaN/Inf; full dtype and implicit/mixed dtype coverage; 0D–8D, non-contiguous/dynamic shapes; constraints and error messages; functional interface; vmap (batch size 8/16/32/64/128); forward/backward with zero deviation; performance gates; bf16/complex as supported.

## 6. Self-Check and Acceptance

- Fill out the CPU operator cross-check checklist item by item (docs/functionality/performance/secure coding); record pass/todo items. Example: mark `mint.cat` as “bf16 pending” in `cat_checklist_validation.md`.

- Run locally many times (e.g., 100 loops) to ensure stability and avoid flakiness. Example: loop the full `test_view.py` suite 100 times.

- In the Feature “Test validation design,” submit the checklist report and representative UT/ST; assign the maintainer for review. Example: attach `test_cat.py` and `test_perf_cat.py` as representative cases in the Feature.

- After clearing issues and archiving deliverables, merge; maintainer closes the Feature to complete. Example: after meeting performance, update the checklist item to “Yes” and merge the PR.

## 7. Troubleshooting

- Build not registered: check exported symbols and paths, CMake logs `Found operator: <Op>`, `MS_OP_PLUGIN_PATH`. Example: `LinSpaceExt` placed in the wrong directory prevented scanning; move to `op_plugin/ops/kernel/linspace_ext.cc` and re-register.
- Numerical mismatch: prefer `*_out`; print error details. Example: `cat` used a non-`_out` interface causing extra allocation; switching to `at::cat_out` aligned results.
- Performance issues: look for extra allocation/synchronization; confirm benchmark scenario and noise removal. Example: missing warm-up in performance test led to >1.1x overhead; adding warm-up fixed it.
- CLA/account: if prompted for CLA, configure `git config --global user.name/email` per FAQ, then resubmit. Example: `git config --global user.email you@huawei.com` resolved the CLA error.

## 8. Development Checklist

- Add `<op>.cc`, `extern "C" int <OpName>(...)`, parse parameters with `KernelInputUtils`.
- Use ATen `*_out` to fill the output tensor.
- Build passes; artifact and `MS_OP_PLUGIN_PATH` configured correctly.
- Functional/performance tests added and included in `tests/run_tests.py`.
- Align with torch: `allclose_nparray` numerics, performance ≤ 1.1x.
- CPU cross-check checklist and secure-coding items done.
- Feature report ready; PR links Feature; awaiting maintainer review/close.
