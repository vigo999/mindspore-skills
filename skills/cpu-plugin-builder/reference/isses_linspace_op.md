### 🚀 背景描述

#### Background

`mint.linspace` 是一类 1D 序列生成算子，用于在区间 `[start, end]` 上按均匀间隔生成指定个数 `steps` 的采样点，常用于构造坐标轴、网格点、插值样本等基础数值计算场景。
本设计文档聚焦在 OP Plugin 中为 **CPU 平台** 适配 `mint.linspace` 接口，对齐 PyTorch `torch.linspace` 语义，并通过 PyBoost 通路在 MindSpore 中复用。

#### Benchmark（参考实现）

- PyTorch:
  - `torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

备注：本方案对标的是 **功能型数值序列生成算子**，不涉及随机性或 inplace 语义。

---

### 功能与接口说明

#### 功能概述

给定标量 `start`、`end` 和整数 `steps`，生成一个一维等差序列张量 `output`：

- `output[0] = start`
- `output[-1] = end`
- 总元素个数为 `steps`，一般约定 `steps >= 2`

数学上，`output` 中第 \(i\) 个元素满足（\(0 \le i < \text{steps}\)）：

\[
\text{output}[i] = \text{start} + i \cdot \frac{\text{end} - \text{start}}{\text{steps} - 1}
\]

#### 对外 functional 接口

MindSpore Python 端通过 YAML `linspace_ext_op.yaml` 自动生成 Primitive/functional，将调用映射到后端的 `linspace` op，并在 `mint` 命名空间下暴露：

```python
mint.linspace(
    start: Union[int, float],
    end: Union[int, float],
    steps: int,
    dtype: Optional[mindspore.dtype] = None,
) -> Tensor  # 1D Tensor, length == steps
```

- `start`：序列起点，标量，支持 int / float（bool 会在前端按标量规则进行提升）；
- `end`：序列终点，标量，支持 int / float；
- `steps`：整型，表示采样点个数，通常要求 `steps >= 2`；
- `dtype`：可选，MindSpore `dtype`，通过 `dtype_to_type_id` 转换为后端 `TypeId`，用于类型推导和输出张量分配。

当前 CPU 适配主要面向 functional 接口，不扩展 Tensor 成员方法。

---

### 任务清单

| 序号 | 任务项             | 任务子项           | 状态（新增/修改/无变更/不涉及） | 备注                                       |
| ---- | ------------------ | ------------------ | ------------------------------- | ------------------------------------------ |
| 1    | 接口基本功能       | Primitive          | 不涉及                          | 由主库 YAML `linspace_ext_op.yaml` 定义   |
|      |                    | functional         | 新增                            | `mint.linspace`                            |
|      |                    | nn                 | 不涉及                          | —                                          |
|      |                    | tensor             | 不涉及                          | —                                          |
| 2    | 后端及数据类型支持 | Ascend             | 无变更                          | 已有 `LinSpaceExtAscend`                  |
|      |                    | GPU                | 不涉及                          | —                                          |
|      |                    | CPU                | 新增                            | 通过 OP Plugin 适配 `LinSpaceExt` CPU 内核 |
| 3    | 支持 vmap          |                    | 不涉及                          | 生成 1D 序列，当前不暴露 vmap 特性        |
| 4    | 支持动态 Shape     | 动态 Shape         | 不涉及                          | 输出长度由 `steps` 决定                   |
|      |                    | 动态 Rank          | 不涉及                          | 输出 Rank 恒为 1                          |
| 5    | 支持反向           | bprop 函数         | 不涉及                          | 作为常量序列生成，一般不对参数求导       |
|      |                    | 复数支持           | 不涉及                          | 以标量浮点为主                             |
| 6    | 补齐资料           | API 映射           | 不涉及                          | —                                          |
|      |                    | 接口中英文资料     | 新增                            | 本文档                                     |
| 7    | 性能优化           | CPU                | 新增                            | 新增 `test_perf_linspace.py` 性能对比用例 |
|      |                    | GPU                | 不涉及                          | —                                          |
|      |                    | Ascend             | 不涉及                          | —                                          |
| 8    | 功能               | 空 Tensor 支持     | 不涉及                          | 输出为 1D，长度由 `steps` 控制            |
|      |                    | inf/nan 支持       | 支持                            | 按浮点运算语义处理                         |
|      |                    | 0~8 维支持         | 不涉及                          | 仅 1D 输出                                 |
|      |                    | 其他功能点         | 无变更                          | —                                          |
| 9    | 门禁用例补齐       | UT                 | 不涉及                          | 主库负责推导与属性检查                     |
|      |                    | ST                 | 新增                            | `test_linspace.py` / `test_perf_linspace.py` |
|      |                    | TEST_OP            | 不涉及                          | —                                          |
| 10   | 支持 MS Adapter    |                    | 不涉及                          | —                                          |
| 11   | 自动并行切分       |                    | 不涉及                          | —                                          |
| 12   | 混合精度（AMP）    |                    | 不涉及                          | 以基础浮点 dtype 为主                      |
| 13   | 安全与异常         | 异常用例与报错规范 | 新增                            | ST 用例中覆盖非法参数场景（后续可补充）   |

---

### 约束与类型

- **设备**：CPU（Linux 平台，通过 OP Plugin 适配）。
- **输入类型**：
  - `start` / `end`：标量，主打 float32 / float64，允许 int 型输入并在前端完成提升；
  - `steps`：int32/int64 整型标量；
  - `dtype`：`TypeId`（前端传入 `mindspore.dtype`）。
- **输出类型**：
  - 输出为 1D Tensor，长度为 `steps`；
  - dtype 优先使用 `dtype` 参数指定，否则由前端根据输入默认推导；
  - 当前 CPU 实现重点保障 float32 / float64 与 PyTorch 对齐，float16 覆盖由 ST 测试校验。
- **范围约束**：
  - 推荐 `steps >= 2`，过小值（如 0 或 1）由前端/主库统一报错或处理；
  - `start` / `end` 可为任意有限浮点数，也可为 `inf`、`nan` 等特殊值，行为与 PyTorch 对齐。

---

### 执行模式与适配

- **Pynative 模式**：
  - Python 端调用 `mint.linspace`；
  - 通过 PyBoost 通路，下发至后端 `linspace` op；
  - 在 CPU 设备上由 OP Plugin 提供的 `LinSpaceExt` kernel 完成实际计算。

- **Graph（KBK/O0）模式**：
  - 通过 `jit(..., backend="ms_backend", jit_level="O0")` 编译执行；
  - 图模式下同样通过 PyBoost 将算子分派到 OP Plugin 的 `LinSpaceExt` 内核；
  - 行为与 Pynative 模式保持一致。

---

### Kernel 设计与实现

#### 1. 参数列表与索引约定

根据 `linspace_ext_op.yaml` 定义，算子参数为：

- `start`：标量输入；
- `end`：标量输入；
- `steps`：整型输入；
- `dtype`：类型参数（`TypeId`），用于前端推导与输出分配；
- `output`：张量输出。

在 OP Plugin kernel 中，仅有输出为张量，其余为标量参数。对应的 C 接口约定为：

```cpp
extern "C" int LinSpaceExt(
  int nparam,
  void **params,
  int *ndims,
  int64_t **shapes,
  const char **dtypes,
  void *stream,
  void *extra);
```

内部使用 `KernelInputUtils` 统一管理非 tensor 参数，索引约定如下：

- `start_idx = 0`：起点标量；
- `end_idx = 1`：终点标量；
- `steps_idx = 2`：步数（整型标量）；
- `dtype` 参数仅参与前端推导，kernel 内不直接访问；
- `tensors[nparam - 1]`：输出张量。

#### 2. 核心计算逻辑

实现文件：`op_plugin/ops/kernel/linspace_ext.cc`，核心逻辑简要如下：

- 使用 `ConvertToATenTensors` 将输出参数转换为 `at::Tensor`；
- 使用 `KernelInputUtils` 从 `KernelInputInfo` 中读取标量参数：
  - `start`、`end` 通过 `GetScalarInput` 获取为 `at::Scalar`；
  - `steps` 通过 `GetIntInput` 获取为 `int64_t`；
- 调用 PyTorch 内核：
  - 使用 `at::linspace_out(output, start, end, steps)` 在已经分配好的输出张量上原位写入结果；
- 返回值：
  - 成功返回 `0`；
  - 不在 kernel 内主动抛出 C++ 异常，错误由底层库或上层框架统一处理。

#### 3. 梯度与反向传播

- `linspace` 主要用于生成常量数值序列，一般不对 `start` / `end` / `steps` 求导；
- 若上层在特定场景需要反向行为，可以依赖框架级 bprop 定义或替代实现，本次 CPU kernel 仅提供 **前向计算**；
- ST 测试中不验证梯度，仅验证前向精度。

---

### 与 PyTorch 的差异与对齐

- **功能对齐**：
  - 输出长度、端点值与 PyTorch `torch.linspace` 保持一致；
  - 支持 `start > end` 的逆向区间，行为与 PyTorch 对齐；
  - 对 `inf` / `nan` 端点，遵循 PyTorch 浮点广播和运算规则。
- **类型与精度**：
  - float32 / float64 精度通过 ST 用例与 PyTorch 一一对比；
  - float16 通过前端转换与 PyTorch 半精度行为对齐。
- **误差度量**：
  - ST 用例中通过 `allclose_nparray` 进行数值对比；
  - 部分场景使用更严格的 `rtol` / `atol`（例如 `2e-8`）验证零偏差或高精度需求。

---

### 测试方案设计

#### 功能用例（`tests/st/mint/test_linspace.py`）

> 说明：功能用例均采用 **单标杆对比法**，以 PyTorch CPU (`torch.linspace`) 为参考，实现 MindSpore 与 PyTorch 的一一对齐。

1. **`test_linspace_std`**
   - Feature：标准功能验证。
   - Description：随机生成 `start`、`end`、`steps`，测试 `dtype=None` 及 `dtype=ms.float32` 场景，在 `pynative` 与 `KBK` 两种模式下，对比 PyTorch 结果。
   - Expectation：输出 shape 与数值均与 `torch.linspace` 一致。

2. **`test_linspace_dtype_coverage`（参数化场景）**
   - Feature：多浮点类型覆盖。
   - Description：对 `ms.float16`、`ms.float32`、`ms.float64` 分别构造随机区间与 steps，验证不同 dtype 下的行为。
   - Expectation：各 dtype 下输出与 `torch.linspace` 一致，误差在 `allclose` 范围内。

3. **`test_linspace_reverse_interval`**
   - Feature：反向区间（`start > end`）支持。
   - Description：构造 `start > end` 的区间，验证递减序列生成是否与 PyTorch 一致。
   - Expectation：输出端点与中间值均对齐 PyTorch。

4. **`test_linspace_numeric_ranges`**
   - Feature：多数值范围覆盖。
   - Description：参数化覆盖如 `(0, 1)`、`(-5, 5)`、`(-1e3, 1e3)`、`(-1e-5, 1e-5)` 等典型数值区间。
   - Expectation：
     - 输出长度为固定 `steps`；
     - 首尾元素分别精确等于 `start`、`end`；
     - 整体序列与 PyTorch 对齐。

5. **`test_linspace_special_values`**
   - Feature：`nan` / `inf` 特殊值处理。
   - Description：覆盖 `(nan, 1.0)`、`(0.0, nan)`、`(inf, 1.0)`、`(-inf, -1.0)` 等组合。
   - Expectation：`nan` / `inf` 传播规则与 PyTorch 一致。

所有功能用例均：

- 使用 `numpy.random` 生成输入参数；
- 通过 `@pytest.mark.parametrize("mode", ["pynative", "KBK"])` 覆盖两种执行模式；
- 使用 `allclose_nparray` 对结果进行 shape + 数值的统一校验。

#### 性能用例（`tests/st/mint/test_perf_linspace.py`）

- Feature：前向性能对比。
- Description：
  - 构造较大规模的 `steps`（如 10 万级）；
  - 在 Pynative 模式下分别循环调用 `mint.linspace` 与 `torch.linspace`，进行 warm-up 后统计多次迭代总耗时；
  - 使用 `BACKGROUND_NOISE` 对环境噪声进行修正。
- Expectation：
  - MindSpore 端到端耗时减去框架底噪之后，不超过 PyTorch CPU 的约 `1.1x`。

---

### 设计思路与代码改动说明

#### 1. 具体算子实现功能直接对标 PyTorch 设计

- 功能语义与 `torch.linspace` 等价；
- 使用 PyTorch C++ 内核 `at::linspace_out` 实现，减少重复造轮子并保证数值一致性。

#### 2. 代码与文件改动说明

- 实现文件：
  - `op_plugin/ops/kernel/linspace_ext.cc`：实现 `LinSpaceExt` CPU kernel，封装 `at::linspace_out` 调用。
- 注册 / YAML：
  - 主库 `linspace_ext_op.yaml` 中的 CPU dispatch 需指向 `LinSpaceExt`（由主库侧配置）。
- 测试：
  - `tests/st/mint/test_linspace.py`：功能泛化测试；
  - `tests/st/mint/test_perf_linspace.py`：性能对比测试。

---

### 测试设计与测试计划

#### 功能验证摘要

| 自测内容                                     | 自测结果 | 备注                                                  |
| -------------------------------------------- | -------- | ----------------------------------------------------- |
| 默认参数场景是否验证                         | 是       | 覆盖 `dtype=None` 与显式 `dtype` 场景                 |
| 空 Tensor 输入正反向是否验证                 | 不涉及   | 输出为 1D，长度由 `steps` 控制                        |
| `inf` / `nan` 是否验证                       | 是       | `test_linspace_special_values`                        |
| 算子支持数据类型是否与标杆对齐               | 是       | 通过 dtype 覆盖用例对齐 float16/32/64                 |
| 输入取值范围是否有验证                       | 是       | 多种数值区间覆盖                                      |
| 输入维度是否覆盖 0D–8D                       | 不涉及   | 输出固定为 1D                                         |
| 输入支持的 dtype 是否全覆盖                  | 是       | 以 float16/32/64 为主                                 |
| 输入是否支持隐式类型转换                     | 部分     | 依赖前端标量提升，ST 不单独验证                       |
| 输入之间的约束是否有验证                     | 是       | 通过 steps、start/end 多组组合间接覆盖                |
| 正向精度验证是否通过                         | 是       | 与 PyTorch CPU 对比通过，采用单标杆对比法             |
| 反向是否支持                                 | 不涉及   | 当前不对参数求导                                      |
| 异常用例是否校验具体报错信息                 | 预留     | 若后续补充，可参照 `test_sign_error_handling` 风格    |
| 是否提供 functional 用例                     | 是       | 所有 ST 均走 `mint.linspace` functional 接口          |
| 动态 shape/rank/属性是否都支持               | 不涉及   | 输出长度由 steps 决定                                 |
| 是否与 PyTorch 计算结果 0 偏差               | 是       | 在给定 `rtol`/`atol` 下通过                           |

#### 性能验证摘要

| 自测内容                                                     | 自测结果 | 备注          |
| ------------------------------------------------------------ | -------- | ------------- |
| 正向端到端耗时减去框架底噪之后的执行时间 ≤ PyTorch CPU 1.1 倍 | 是       | `test_perf_linspace.py` |

---