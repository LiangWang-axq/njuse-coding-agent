# 设计说明

## 目标

构建一个不依赖任何 Agent 框架/SDK 的命令行编程智能体：通过大语言模型自主读写文件、执行受控命令，完成编程任务。核心逻辑全部自研：对话历史与上下文管理、工具定义与本地执行、模型输出解析、循环终止、错误处理。

## 总体架构

```text
cli.py -> repl.py（交互循环）或 CodingAgent.run() -> ContextManager(history)
          -> OpenAICompatibleProvider.chat()/chat_stream()  (原生 tools / JSON / SSE 流式)
          -> parse_model_response()           (动作解析)
          -> ToolRegistry.execute()           (本地工具，工作区锁定)
          -> 工具结果回填 history -> 循环，直到 final 或达到 max_steps
```

模块职责：

- `coding_agent/agent.py`：主循环 `CodingAgent.run`，负责调用模型、解析输出、执行工具、回填结果、计数终止条件；`on_step` 回调把每一步工具调用暴露给 CLI 展示。
- `coding_agent/repl.py`、`coding_agent/ui.py`：交互式对话循环（斜杠命令、会话管理、Ctrl+C/EOF 处理、多轮历史复用）与纯标准库 ANSI 渲染（TTY 检测、Windows VT 开启、工具调用和会话列表格式化）。
- `coding_agent/context.py`：`ContextManager` 维护完整消息历史；`prepare()` 输出非破坏压缩视图（四层廉价优先：L3 大结果落盘、L1 中间裁切、L2 旧结果摘要、L4 LLM 结构化摘要），并用真实 usage 校准 token 估算，保证模型输入窗口可控。
- `coding_agent/session.py`：`Session` 把每条消息原子落盘为 JSONL（header + 消息行）；`SessionInfo` 与扫描、选择、删除接口支持按序号、完整 ID 或唯一前缀管理会话，损坏文件仍可见且可删除。
- `coding_agent/protocol.py`：`parse_model_response` 兼容两种协议：JSON 动作（`{"type":"tool_call"|"final"}`）与 OpenAI 原生 `tool_calls`；支持一次多个工具调用依序执行；连续两次解析失败即终止。
- `coding_agent/provider.py`：极简 HTTP 客户端，仅走 Chat Completions 接口；`chat_stream` 自行解析 SSE（按行缓冲 `data:` 事件、`[DONE]` 终止、原生 `tool_calls` 按 index 增量拼接参数）；429/5xx/超时按 2s/4s 退避重试；网关拒绝 `tools` 参数（HTTP 400）时自动降级为纯 JSON 协议。
- `coding_agent/tools.py`：10 个本地工具（list/read/search/write/replace/apply_diff/run_tests/delete/move/git_status），全部通过 `WorkspaceTools._path` 做工作区根路径校验；`run_tests` 支持 `cwd` 参数在指定的工作区子目录下执行。
- `coding_agent/config.py`、`cli.py`、`workspace.py`：配置加载、工作区解析与命令行入口。

## 关键设计决策

### 为什么不用 Agent 框架

题目要求重要逻辑自行编写。框架会把历史管理、工具循环、终止判断封装成黑盒，既无法体现设计，答辩也难以展开。本项目每个环节都是显式、可测试的模块。

### 双协议输出解析

主流模型支持原生 `tool_calls`，但部分 OpenAI 兼容网关不支持 `tools` 参数。因此请求默认带 `tools`，解析层同时支持原生调用与纯 JSON 动作；网关报 400 时降级为 JSON 协议，兼容性最好。

### 流式输出与增量工具调用

交互模式与一次性模式都走 `chat_stream`：SSE 按行缓冲，文本 `delta` 直接回调终端逐字显示；原生 `tool_calls` 在流中按 `index` 累加 `id/name/arguments` 字符串，结束时 `json.loads` 校验后交给与普通请求相同的 `parse_model_response`。重试只发生在首块数据到达前，避免中途重复输出。

### 多轮会话与终端交互

`CodingAgent` 持有同一个 `ContextManager`，`run()` 首次调用创建、后续调用复用，因此交互模式下多轮任务共享历史；`/new` 调用 `reset()` 清空，`/resume` 调用 `switch_session()` 重放所选历史并重置会话级运行统计。界面仅用标准库 ANSI 颜色：`ctypes` 开启 Windows VT 处理，非 TTY 或设置 `NO_COLOR` 时自动去色，保持零第三方依赖。

### 工作区选择

CLI 默认将当前目录作为工作区，也可通过 `--workspace PATH` / `-w PATH` 指定已有目录。REPL 的 `/workspace [路径]` 支持查看并切换工作区：切换路径后重新创建 `CodingAgent`，因此工具路径校验、模型配置、会话目录和上下文都绑定到新目录；原工作区的会话不被带入，切换后可在新工作区使用 `/sessions` 与 `/resume`。路径解析只要求目标是已有目录，工作区本身可以位于当前目录之外。

### 上下文压缩策略

模型输入有长度限制，长任务会把早期步骤挤出窗口。`ContextManager` 保存完整历史，每次调用模型前由 `prepare()` 生成压缩视图（非破坏，真实历史 100% 保留，供会话持久化与恢复）：

- **L3 大结果落盘**：超过 12000 字符的工具结果先写入 `<工作区>/.coding_agent/results/`，视图换成“落盘路径 + 4000 字符预览”，防超大单条撑爆窗口；
- **L1 中间轮次裁切**：消息数超过 60 条时保留头部与最近尾部，中间插占位符；
- **L2 旧结果占位**：仍超预算时，把较早的工具结果压成一行摘要（工具名 + 成功/失败 + 关键字段，stdout/stderr 只留长度），最近 2 条不动，0 API 损耗；
- **L4 LLM 结构化摘要**：前三层后仍超预算才调用模型，用 `<analysis>` / `<summary>` 双标签生成结构化摘要（防注入，只剥离 `<summary>` 正文），失败时降级不压缩；
- **Usage 锚定**：默认按 `字符数 / 4` 估算 token；每轮模型返回真实 `usage.prompt_tokens` 后校准比例（provider 已在非流式/流式两种路径捕获 usage），估算精度随对话推进提升。

这样模型始终知道“之前做了什么”，且只有 L4 在最坏情况下花 1 次额外 API。

### 界面可视化

终端体验围绕“可讲解”做了三层可视化：

- **工具多行卡片**：每次工具调用渲染为“工具名 + 参数 + 结果摘要”卡片；`read_file` 显示路径与字符数、`run_tests` 显示通过/失败并附输出尾部、`apply_diff` 显示 hunk 与增删行数，失败时红色标注错误；
- **压缩事件实时播报**：`ContextManager.prepare` 记录本次触发过的层（L3 落盘 N 条 / L1 裁切 N 条 / L2 压缩 N 条 / L4 摘要 / 兜底截断），`Agent.run` 通过 `on_context` 回调在压缩状态变化时打印 `[上下文] …`；相同压缩状态不重复刷屏；
- **Token 用量与预算**：每轮结束显示 `token prompt X / completion Y`（有真实 usage 时）或估算值，并标注上下文预算；`/status` 额外展示会话累计压缩统计与最近一次 usage。

这些展示全部由纯标准库 ANSI 完成，管道输出自动去色，不引入任何 UI 依赖。

### 会话持久化

`CodingAgent` 构造时在 `<工作区>/.coding_agent/sessions/` 下创建 JSONL 会话文件（文件名带时间戳，天然有序）。每追加一条消息（任务、助手输出、工具结果、纠错反馈）都同步 `Session.add` 原子落盘：临时文件写入 + `fsync` 强制刷盘 + `os.replace` 原子替换，断电或杀进程不会损坏上次快照；加载时若检测到崩溃残留的撕裂尾行，直接丢弃该行。

`--resume` 启动时加载工作区内最近一次会话，`--resume-session` 恢复指定会话，二者都会把历史消息重放进 `ContextManager`。`--list-sessions` 与 `/sessions` 按创建时间最新优先展示序号、状态、ID、创建时间、消息数和首条用户消息摘要；损坏文件明确标记且禁止恢复。

CLI 的 `--delete-session` 与交互命令 `/delete` 在确认后永久删除选中的 JSONL；删除当前会话后立即 `reset()`，删除其他会话不触碰当前上下文。选择和删除始终基于工作区会话目录的扫描结果，不直接把用户输入解释为文件路径。`.coding_agent/results/` 目前没有会话归属索引，因此删除会话不连带清理大结果文件。

### 安全边界

- 所有文件操作路径先解析再校验，必须是工作区内的相对路径；拒绝绝对路径、盘符、`..` 与 `.env`。
- `run_tests` 不经过 shell，仅允许 `python -m unittest/pytest/compileall`，禁止 `;`、`&&`、管道等拼接，120 秒超时；可用 `cwd` 指定工作区内相对目录，仍不越界。
- `git_status` 只读且参数固定，无 shell 拼接。
- API key 只从环境变量或未入库的 `.env` 读取，工具层禁止读写 `.env`。

### 终止与错误处理

- 正常终止：模型返回 `final`。
- 兜底终止：达到 `max_steps`（默认 24）。
- 解析终止：连续两次格式错误。
- 错误恢复：工具执行失败（路径不存在、参数错误等）作为带错误信息的工具结果回传模型，让模型自己修正，而不是直接崩溃。
- 网络错误：429/5xx/超时退避重试，其余 4xx 快速失败并给出可读信息。
- 收敛规则：系统提示固定要求“测试通过后立即返回 final、不重复读取已确认文件、验证语法用 compileall 而非 python -c”，防止模型陷入“重复确认”的无效循环。

### diff 级编辑

`apply_diff` 用标准 unified diff 精准修改单个已有文本文件：先解析 `@@` hunk（空格=上下文、`-`=删除、`+`=新增），按顺序逐 hunk 校验上下文与删除行是否逐字一致，任一 hunk 不匹配就整次失败且不写盘（原子应用）；定位时优先使用 `@@` 声明的位置，若声明位置不匹配则只接受文件中唯一无歧义的匹配，避免误改；hunk 头缺行号（如只有 `@@`）时自动退化为“全文件唯一匹配”定位，方便兼容模型生成的不完整 diff。新增/删除文件仍走 `write_file` / `delete_file`，保持职责单一。

## 已知限制与改进方向

- 会话管理限定在当前工作区，尚未提供跨工作区的全局会话索引；大结果文件也尚未按会话隔离存储。
- 已用 usage 校准上下文估算；尚未在界面上展示累计 token/成本统计。
- 工具串行执行；可对只读工具做并行化。
