# JiuwenClaw + accuracy-agent 技能使用示例

## 在服务器安装与启动 jiuwenclaw 服务

首先，在服务器上安装 jiuwenclaw，安装步骤请参考官方文档：[快速开始 \| JiuwenClaw](https://openjiuwen.com/docs-page?version=jiuwenclaw-v0.1.7-zh-ypJXJA0s&path=%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)

![](assets/Pasted%20image%2020260405180030.png)

安装完成后，需分别启动 Web 前端服务和后端服务。

在服务器上打开一个终端窗口，执行以下命令启动 jiuwenclaw 的 Web 前端服务。`--port` 参数可根据实际需要指定端口号。

```bash
jiuwenclaw-web --host 0.0.0.0 --port 5173
```

![](assets/Pasted%20image%2020260405180510.png)

随后，在服务器上另开一个终端窗口，执行以下命令启动 jiuwenclaw 后端服务。

```bash
jiuwenclaw-app
```

![](assets/Pasted%20image%2020260405180839.png)

至此，服务端配置完成。

在本地浏览器中，通过 `http://<服务器IP>:<端口号>` 访问 jiuwenclaw Web 工作台：

![](assets/Pasted%20image%2020260405181300.png)

发送一条测试消息，若能正常收到回复，则说明前后端服务均已正常运行。

![](assets/Pasted%20image%2020260405181527.png)

---

## 安装 mindspore-skills

jiuwenclaw 通过技能系统扩展 Agent 的能力，关于技能的安装与管理可参考官方文档：[技能系统 \| JiuwenClaw](https://openjiuwen.com/docs-page?version=jiuwenclaw-v0.1.7-zh-ypJXJA0s&path=%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%2F%E6%8A%80%E8%83%BD%E7%B3%BB%E7%BB%9F)

在左侧导航栏点击"技能"进入技能管理面板，然后点击"源管理"。

![](assets/Pasted%20image%2020260405181816.png)

添加 mindspore-skills 的 GitHub 仓库链接：[mindspore-skills](https://github.com/mindspore-lab/mindspore-skills)

![](assets/Pasted%20image%2020260405181853.png)

添加完成后，该技能源默认处于禁用状态。在"已配置源"列表中找到 mindspore-skills，点击"启用"按钮将其激活。

![](assets/Pasted%20image%2020260405182016.png)

此时技能面板中将显示 mindspore-skills 仓库内的所有技能，状态为"未安装"。

本示例需要安装 accuracy-agent 技能，找到该技能并点击右侧的"安装"按钮。

![](assets/Pasted%20image%2020260405182125.png)

点击"安装"后，浏览器将弹出输入框，要求确认插件规格（格式为 `技能名@源名`，如 `accuracy-agent@mindspore-skills`），点击"确定"即可。

![](assets/Pasted%20image%2020260405182257.png)

安装完成后，技能状态将更新为"已安装"。

![](assets/Pasted%20image%2020260405182524.png)

可在服务器上验证安装结果，已安装的技能文件位于 `~/.jiuwenclaw/agent/skills` 目录下。

![](assets/Pasted%20image%2020260405182741.png)

---

## 使用 mindspore-skills 进行精度诊断

以下演示使用 mindspore-skills 中的 accuracy-agent 技能（精度问题诊断助手）定位并修复一个模型精度问题。

在 Web 工作台中选择"智能执行"模式，输入如下诊断指令并发送：

```text
去 exp4-accuracy-agent/mscli-demos 中查看一下，@run_llm_infer.sh 运行了同一个模型的 torch_npu 和 mindspore 版本推理脚本，但是输出的结果有精度误差。
- 预期精度应该是绝对的0偏差对齐，不允许有微小误差。
- 定位并修复这个问题。
- 使用 mscli-demo-accuracy 这个 conda 环境中的 python 来运行所有脚本。
```

![](assets/Pasted%20image%2020260405183115.png)

jiuwenclaw 自动调用 accuracy-agent 技能执行诊断，首先识别任务类型，规划诊断步骤并开始查看相关文件：

![](assets/Pasted%20image%2020260405183237.png)

通过分析代码，定位到关键问题：`LayerNorm` 的 epsilon 默认值在不同框架间存在差异（`mindspore.nn.LayerNorm` 默认 `epsilon=1e-7`，而 `torch.nn.LayerNorm` 和 `mindspore.mint.nn.LayerNorm` 默认 `eps=1e-5`）：

![](assets/Pasted%20image%2020260405183624.png)

构建最小复现脚本进行对比实验，确认根因：

![](assets/Pasted%20image%2020260405183859.png)

提出两种修复方案，并根据 accuracy-agent 技能的指导选择使用 `mint.nn.LayerNorm` 替换 `nn.LayerNorm`：

![](assets/Pasted%20image%2020260405183947.png)

生成诊断报告，包含症状、根因分析和框架间 API 对比：

![](assets/Pasted%20image%2020260405184308.png)

展示具体的代码修复内容（修改 `infer_llm_ms.py` 第 63、68 行）：

![](assets/Pasted%20image%2020260405184336.png)

修复后重新运行验证，所有输出完全对齐，达到绝对 0 偏差：

![](assets/Pasted%20image%2020260405184350.png)

以上验证了 mindspore-skills 在 jiuwenclaw 中的完整运行流程。
