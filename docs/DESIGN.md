# 设计说明

## 目标

构建一个不依赖任何 Agent 框架/SDK 的命令行编程智能体：通过大语言模型自主读写文件、执行受控命令，完成编程任务。核心逻辑全部自研：对话历史与上下文管理、工具定义与本地执行、模型输出解析、循环终止、错误处理。

## 总体架构

```text
cli.py -> CodingAgent.run() -> ContextManager(history)
          -> OpenAICompatibleProvider.chat()  (原生 tools / JSON 协议)
          -> parse_model_response()           (动作解析)
          -> ToolRegistry.execute()           (本地工具，工作区锁定)
          -> 工具结果回填 history -> 循环，直到 final 或达到 max_steps
```

模块职责：

- `coding_agent/agent.py`：主循环 `CodingAgent.run`，负责调用模型、解析输出、执行工具、回填结果、计数终止条件；`on_step` 回调把每一步工具调用暴露给 CLI 展示。
- `coding_agent/context.py`：`ContextManager` 维护消息历史，超限时丢弃最旧工具结果对并插入压缩摘要，保证模型输入窗口可控。
- `coding_agent/protocol.py`：`parse_model_response` 兼容两种协议：JSON 动作（`{"type":"tool_call"|"final"}`）与 OpenAI 原生 `tool_calls`；支持一次多个工具调用依序执行；连续两次解析失败即终止。
- `coding_agent/provider.py`：极简 HTTP 客户端，仅走 Chat Completions 接口；429/5xx/超时按 2s/4s 退避重试；网关拒绝 `tools` 参数（HTTP 400）时自动降级为纯 JSON 协议。
- `coding_agent/tools.py`：9 个本地工具（list/read/search/write/replace/run_tests/delete/move/git_status），全部通过 `WorkspaceTools._path` 做工作区根路径校验。
- `coding_agent/config.py`、`cli.py`：配置加载与命令行入口。

## 关键设计决策

### 为什么不用 Agent 框架

题目要求重要逻辑自行编写。框架会把历史管理、工具循环、终止判断封装成黑盒，既无法体现设计，答辩也难以展开。本项目每个环节都是显式、可测试的模块。

### 双协议输出解析

主流模型支持原生 `tool_calls`，但部分 OpenAI 兼容网关不支持 `tools` 参数。因此请求默认带 `tools`，解析层同时支持原生调用与纯 JSON 动作；网关报 400 时降级为 JSON 协议，兼容性最好。

### 上下文压缩策略

模型输入有长度限制，长任务会把早期步骤挤出窗口。`ContextManager` 以字符预算（默认 16000）为界：超限时丢弃最旧消息对，把已完成的工具调用压缩成一条摘要（工具名 + 结果简述）插入历史顶部；单条超大消息最后兜底截断。这样模型始终知道“之前做了什么”。

### 安全边界

- 所有文件操作路径先解析再校验，必须是工作区内的相对路径；拒绝绝对路径、盘符、`..` 与 `.env`。
- `run_tests` 不经过 shell，仅允许 `python -m unittest/pytest/compileall`，禁止 `;`、`&&`、管道等拼接，120 秒超时。
- `git_status` 只读且参数固定，无 shell 拼接。
- API key 只从环境变量或未入库的 `.env` 读取，工具层禁止读写 `.env`。

### 终止与错误处理

- 正常终止：模型返回 `final`。
- 兜底终止：达到 `max_steps`（默认 12）。
- 解析终止：连续两次格式错误。
- 错误恢复：工具执行失败（路径不存在、参数错误等）作为带错误信息的工具结果回传模型，让模型自己修正，而不是直接崩溃。
- 网络错误：429/5xx/超时退避重试，其余 4xx 快速失败并给出可读信息。

## 已知限制与改进方向

- 目前是整体写入 + 精确替换，无 diff 级编辑；可增加 `apply_diff` 工具。
- 会话不可断点续跑；可把 history 序列化保存，支持 `--resume`。
- 无 token/成本统计；可在 provider 返回 usage 并汇总。
- 工具串行执行；可对只读工具做并行化。
