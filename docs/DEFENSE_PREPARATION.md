# Coding Agent 设计方案与答辩准备

> 用途：准备项目介绍、原理讲解和评委问答。
>
> 使用方式：后续根据实际演示、评委问题和个人表达习惯持续修改润色。

## 一、项目定位

本项目是一个不依赖 LangChain、AutoGen 等 Agent 框架的命令行编程智能体。

它的目标不是让大语言模型直接控制电脑，而是让模型在一个受控工作区内，通过有限工具完成“分析、修改、验证”的编程闭环。

项目的核心思想是：

> 大语言模型负责决策，本地程序负责执行和约束。

模型不能直接访问文件系统，也不能直接运行命令。它只能看到当前上下文，并从预先定义好的工具中选择下一步动作；真正的文件读写、测试执行和安全检查由本地 Python 程序完成。

## 二、整体运行原理

整体流程如下：

```text
用户输入任务
    ↓
构造 system prompt、工具说明和历史消息
    ↓
调用大语言模型
    ↓
解析模型输出
    ↓
如果是工具调用：执行本地工具
    ↓
把工具结果加入对话历史
    ↓
重新调用模型
    ↓
模型返回 final 或达到终止条件
```

主循环位于 `coding_agent/agent.py` 的 `CodingAgent.run()`。

它的逻辑可以简化为：

```python
append(user_task)

for step in range(max_steps):
    view = context.prepare()
    raw = provider.chat_or_stream(view)
    append(assistant, raw)

    response = parse_model_response(raw)

    if response.kind == "final":
        return success

    result = execute_tool(response)
    append(user, tool_result)

return failure
```

例如用户输入：

```text
修复订单模块的金额计算 bug，并运行测试
```

Agent 可能依次执行：

```text
1. list_files
2. read_file
3. run_tests
4. 根据失败断言定位问题
5. apply_diff 或 replace_in_file
6. 再次 run_tests
7. 测试通过
8. 返回 final
```

这里的“自主性”不是模型一次性生成全部代码，而是模型根据每一步工具返回的真实环境反馈，动态决定下一步做什么。

## 三、对话历史与上下文管理

### 3.1 完整历史

`ContextManager` 内部维护类似下面的消息列表：

```python
[
    {"role": "system", "content": "系统规则和工具说明"},
    {"role": "user", "content": "用户任务"},
    {"role": "assistant", "content": "模型动作"},
    {"role": "user", "content": "工具结果"},
]
```

每次模型调用前执行：

```python
view = context.prepare()
```

`prepare()` 不会删除或修改真实历史，而是生成一个发送给模型的压缩视图。完整历史同时保存在内存和 JSONL 会话文件中。

### 3.2 为什么需要上下文压缩

编程任务进行多轮后，会产生大量文件内容、测试输出、错误堆栈、模型动作和工具结果。如果全部发送给模型，可能造成：

- 超过模型上下文窗口；
- Token 成本增加；
- 请求变慢或失败；
- 早期的重要信息被挤出上下文。

因此项目使用了四层上下文压缩管线。

### 3.3 四层压缩机制

#### L3：大工具结果落盘

当工具结果超过 12000 个字符时，将完整结果写入：

```text
.coding_agent/results/
```

发送给模型的内容只保留落盘路径和前 4000 个字符的预览。

这样既保存了完整信息，也避免一条超大消息撑爆上下文。

#### L1：裁切中间消息

当消息数量超过 60 条时，保留 system 消息、任务开头和最近消息，用占位符替代中间历史。

原因是最近的测试结果和当前状态通常比很早以前的重复读取更重要。

#### L2：旧工具结果摘要

如果仍然超出预算，就把较早的工具结果压缩成一行摘要，例如：

```text
工具结果 read_file：成功 {"path":"orders.py","content":"<2340 字符已省略>"}
```

最近两条工具结果不压缩，确保模型仍能看到最新状态。

#### L4：调用模型总结历史

前三层压缩后仍然超限时，才额外调用一次模型，把旧历史整理为：

```text
## 目标
## 进度
## 关键决定
## 下一步
```

摘要请求要求模型将旧对话视为数据，而不是新的指令，从而降低历史内容中的指令注入风险。

### 3.4 Token 估算与校准

初始估算采用：

```text
字符数 / 4 ≈ Token 数
```

每次模型返回真实的 `prompt_tokens` 后，系统根据实际 Token 数和本次字符数计算比例，用于校准后续估算。因此上下文预算不是完全固定的经验值，而是会随着实际调用结果逐步修正。

## 四、工具定义与本地执行

### 4.1 工具定义

工具定义位于 `coding_agent/tools.py` 的 `tool_schemas()`，包含工具名称、用途、参数名称、参数类型和必填参数。

当前共有 10 个工具：

- `list_files`：列出工作区文件；
- `read_file`：读取 UTF-8 文本文件；
- `search_code`：搜索代码或文本；
- `write_file`：创建或整体写入文件；
- `replace_in_file`：按精确文本替换；
- `apply_diff`：按 unified diff 精确修改文件；
- `run_tests`：运行受控测试命令；
- `delete_file`：删除文件；
- `move_file`：移动或重命名文件；
- `git_status`：查看 Git 状态和最近提交。

模型看到的是结构化 Schema，例如：

```json
{
  "name": "read_file",
  "description": "读取工作区内 UTF-8 文本文件",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"]
  }
}
```

### 4.2 工具注册表

`ToolRegistry` 将模型可以使用的工具名映射到本地 Python 方法：

```python
self._tools = {
    "list_files": impl.list_files,
    "read_file": impl.read_file,
    "search_code": impl.search_code,
    "write_file": impl.write_file,
    "replace_in_file": impl.replace_in_file,
    "apply_diff": impl.apply_diff,
    "run_tests": impl.run_tests,
    "delete_file": impl.delete_file,
    "move_file": impl.move_file,
    "git_status": impl.git_status,
}
```

模型只能请求注册表中存在的工具。未知工具会被拒绝。

### 4.3 为什么工具在本地执行

模型只输出类似下面的动作：

```json
{
  "type": "tool_call",
  "tool": "read_file",
  "arguments": {
    "path": "orders.py"
  }
}
```

真正的执行由本地代码完成。这样有三个好处：

1. 模型不能直接控制操作系统；
2. 所有路径和参数可以统一校验；
3. 文件修改和测试执行过程可记录、可测试、可复现。

### 4.4 路径安全

所有文件操作都会经过 `_path()` 校验。路径解析后必须位于工作区根目录之内，并拒绝：

- 绝对路径；
- Windows 盘符路径；
- UNC 路径；
- `..` 越界路径；
- `.env` 文件。

因此 Agent 的工作范围被锁定在启动时指定的工作区内。

### 4.5 文件修改方式

项目提供三种主要修改方式：

- `write_file`：适合新建文件或确实需要整体重写的文件；
- `replace_in_file`：适合已知原文的精确替换；
- `apply_diff`：适合对已有代码做局部修改。

`replace_in_file` 会检查实际替换次数。如果期望替换 1 次，但实际找到 0 次或多次，就拒绝修改，避免误改。

`apply_diff` 会先验证所有 hunk 的上下文和删除内容，全部匹配后才写入。它具有逻辑上的“失败不写入”特性，但不是断电场景下的磁盘级原子写入。

### 4.6 测试命令安全

`run_tests` 使用 `subprocess.run(..., shell=False)`，只允许：

```text
python -m unittest ...
python -m pytest ...
python -m compileall ...
```

同时拒绝 Shell 拼接、管道、重定向和脚本展开等字符。因此模型不能通过测试工具拼接任意 Shell 命令。

## 五、模型输出解析

解析逻辑位于 `coding_agent/protocol.py` 的 `parse_model_response()`。

### 5.1 普通 JSON 动作

工具调用：

```json
{
  "type": "tool_call",
  "tool": "run_tests",
  "arguments": {
    "command": "python -m unittest discover -s tests -v"
  }
}
```

任务完成：

```json
{
  "type": "final",
  "message": "测试已全部通过"
}
```

### 5.2 OpenAI 原生 tool calls

项目也支持：

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"orders.py\"}"
      }
    }
  ]
}
```

解析器会将两种协议统一转换成 `ParsedResponse`，主循环不需要关心模型使用了哪种协议。

### 5.3 容错解析

解析器支持：

- Markdown 代码块中的 JSON；
- JSON 前后附带少量文字；
- 工具参数是 JSON 字符串的情况；
- 一次返回多个工具调用。

多个工具调用会按照返回顺序依次执行，最多处理 8 个，防止异常响应扩大执行量。

### 5.4 为什么需要双协议

并不是所有 OpenAI 兼容网关都支持 `tools` 参数。Provider 默认请求原生工具调用；如果网关返回 HTTP 400，系统会自动关闭原生工具参数，降级为普通 JSON 协议。

这样既保留了原生 tool calling 的结构化能力，又提高了不同模型服务之间的兼容性。

## 六、循环终止条件

### 6.1 正常终止

模型返回 `final` 后，主循环立即返回成功结果。

### 6.2 最大步骤数终止

默认最多执行 24 次模型-工具循环。达到上限后返回“任务尚未确认完成”，避免模型无限运行。

### 6.3 连续解析失败终止

第一次输出无法解析时，将格式错误反馈给模型；第二次仍然失败时，终止任务。

工具执行失败不会立即终止，因为路径或参数错误可能可以由模型修正。

### 6.4 当前实现的边界

目前系统主要通过 system prompt 要求“测试通过后返回 final”，并没有在运行时强制检查模型返回 final 前是否执行过测试。

后续可以增加状态机，例如要求至少存在一次成功的测试结果后才接受 final。但这会降低自由任务的灵活性，因此当前版本采用了“提示约束 + 测试反馈 + 最大步骤数”的折中方案。

## 七、错误处理机制

### 7.1 工具错误

工具错误会被转换成结构化结果：

```json
{
  "ok": false,
  "tool": "read_file",
  "error": "文件不存在: xxx.py"
}
```

然后回填给模型，让模型修正参数或调整方案。

### 7.2 API 错误

Provider 会对 429、500、502、503、504 和请求超时进行退避重试，间隔为 2 秒、4 秒。

其他 4xx 错误直接失败，因为这类错误通常是认证、参数或权限问题，重复请求没有意义。

### 7.3 流式响应错误

流式请求使用 SSE：

- 文本增量直接显示；
- tool call 参数按照 index 累加；
- 收到 `[DONE]` 后统一解析。

只有在第一个流式事件到达之前才允许重试，避免已经输出一部分内容后重新输出，导致终端内容重复。

### 7.4 会话错误

每条历史消息都会写入 JSONL 文件。保存过程使用临时文件、`fsync` 和 `os.replace`。

如果程序崩溃时最后一行只有半截 JSON，恢复时会丢弃该行，保留之前有效的消息。

## 八、为什么不使用 Agent 框架

题目要求重要逻辑自行实现。Agent 的关键不只是调用模型，还包括：

- 对话历史维护；
- 上下文压缩；
- 工具定义和执行；
- 输出协议解析；
- 错误恢复；
- 循环终止；
- 会话持久化。

如果直接使用框架，这些机制容易被封装成黑盒，不利于说明设计和验证设计。因此项目只使用 OpenAI 兼容接口，其他关键环节全部自行实现。

## 九、项目验证情况

当前测试覆盖：

- Agent 主循环；
- JSON 和原生 tool call 解析；
- 工具路径安全；
- diff 修改；
- 测试命令白名单；
- 上下文四层压缩；
- 会话保存和恢复；
- 流式输出；
- API 重试和协议降级；
- 终端交互。

已验证：

```text
测试套件：91 项全部通过
离线演示：成功完成 Todo 删除 Bug 的测试、修改和复验，共 8 步
```

答辩时需要注意：`docs/DEMO.md` 推荐的是订单模块 `fix_me` 演示，而 `demo/run_offline_demo.py` 实际演示的是 Todo 项目。应根据现场录屏内容选择对应说法。

## 十、English Defense Version

### 10.1 One-minute project introduction

> My project is a framework-free command-line Coding Agent. Its purpose is to help users complete real programming tasks, such as inspecting files, fixing bugs, and running tests. The core idea is simple: the language model makes decisions, while local Python code executes actions and enforces safety rules.
>
> The Agent follows a ReAct-style loop: it observes the task and the current context, selects a tool, receives the tool result, and then decides what to do next. The available tools can list files, read and search code, edit files, run tests, and inspect Git status. All paths are restricted to the workspace, and test commands use a whitelist without shell composition.
>
> I implemented the main loop, output parsing, context compression, session persistence, error recovery, and termination rules myself. This makes the system easier to understand, test, and defend than a framework-based black box.

### 10.2 Possible questions and answers

#### Q1: Is your system using the ReAct pattern?

**Answer:** Yes. It is a simplified and implicit ReAct loop. The model decides the next action, the local tool executes it, and the result is returned as an observation for the next decision. I do not expose or save the model's full chain of thought; I only keep structured actions, tool results, and the final answer.

#### Q2: Why does the model need tools instead of directly generating code?

**Answer:** Code generation alone cannot verify the actual workspace state. The model needs to inspect real files, run real tests, and react to real errors. Tools connect the model's decisions with the local environment, turning a text response into an executable programming workflow.

#### Q3: Who is responsible for execution: the model or the Agent?

**Answer:** The model is responsible for selecting the next action, but the local Agent is responsible for execution. The model only returns a tool name and arguments. The local registry checks the tool, validates the arguments, and calls the corresponding Python function.

#### Q4: Can the model execute arbitrary commands?

**Answer:** No. The model can only request registered tools. File paths must stay inside the workspace, and the test tool only accepts Python unittest, pytest, or compileall commands. It also uses `shell=False` and rejects shell chaining, pipes, and redirection. This is a workspace-level safety boundary, not a complete operating-system sandbox.

#### Q5: Why do you support both JSON actions and native tool calls?

**Answer:** Different OpenAI-compatible gateways support different formats. Native tool calls are more structured, but some gateways reject the `tools` parameter. Therefore, the parser supports native tool calls and a plain JSON action protocol, and the provider can fall back to JSON when native tools are rejected.

#### Q6: Why can one response contain multiple tool calls?

**Answer:** The API represents tool calls as a list, and the model may decide that several independent read operations can be requested in one response. I convert them into an ordered list and execute them sequentially. They are not executed in parallel in the current version, which avoids file-state conflicts and keeps the behavior deterministic.

#### Q7: Why are tool results stored as `user` messages?

**Answer:** Semantically, they are observations from the environment. I store them as `user` messages because some compatible gateways do not support the native `tool` role. This gives the JSON protocol and native tool-call protocol one common history format.

#### Q8: Why do you need context compression?

**Answer:** A coding task may produce many file contents and test outputs, so sending the complete history can exceed the model's context limit. My system preserves the full history but creates a compressed view for each request. Large results are stored on disk, old results are summarized, and an additional model summary is used only when the cheaper steps are not enough.

#### Q9: How does the Agent know when to stop?

**Answer:** It stops normally when the model returns a `final` action. It also stops after a maximum number of steps or after two consecutive parsing failures. Tool errors are returned to the model so it can correct its action instead of stopping immediately.

#### Q10: Why do you not enforce a successful test before accepting `final`?

**Answer:** The current version uses system instructions and test feedback to guide the model, together with a maximum-step safeguard. It does not yet use a strict state machine to reject every `final` without a previous successful test. This keeps the Agent flexible for tasks that do not require tests, but a future version could add task-dependent verification states.

#### Q11: Is this a multi-agent system?

**Answer:** No. It is a single Agent with multiple local tools. The current task requires one consistent context while inspecting, editing, and testing the same workspace. A multi-agent design would add coordination and file-conflict problems without clear benefits for this scope.

#### Q12: Is the project only prompt engineering?

**Answer:** No. The prompt describes the model's role and output rules, but the executable behavior comes from the local main loop, tool registry, path validation, protocol parser, context manager, session storage, retry logic, and termination rules. These components are independently testable.

#### Q13: How do you prove that the design works?

**Answer:** The tests cover the main loop, both output protocols, tool safety, diff editing, context compression, session recovery, streaming, API retries, and terminal interaction. The current test suite has 91 passing tests, and the offline demo completes a test-failure, code-edit, and test-success workflow.

#### Q14: What are the current limitations?

**Answer:** Tool calls are currently sequential, session management has only basic resume support, and final verification is mainly guided by the prompt rather than enforced by a strict state machine. Future improvements could include parallel read-only tools, stronger verification states, session management commands, and container-level isolation.

## 十一、评委可能提问及回答

### 问题 1：你的 Agent 和普通聊天机器人有什么区别？

普通聊天机器人主要生成文本，而我的 Agent 能根据模型决策调用本地工具，并观察工具反馈继续行动。它形成了“决策—执行—观察—再决策”的闭环，所以不仅能回答问题，还能实际修改代码和运行测试。

### 问题 2：模型为什么知道什么时候读取文件、什么时候运行测试？

模型通过工具 Schema 了解每个工具的功能和参数，通过 system prompt 了解任务流程。工具返回实际文件内容或测试结果后，模型再根据这些证据选择下一步。

但最终工具能否执行，不由模型决定，而由本地注册表和参数校验决定。

### 问题 3：模型会不会直接执行任意命令？

不会。模型只能请求注册过的工具，工具层还会再次校验参数。`run_tests` 只允许规定的 Python 测试和语法检查命令，并且使用 `shell=False` 执行。

不过这个项目不是完整的操作系统级沙箱。它主要解决工作区越界和 Shell 注入问题；面对完全恶意的代码，还需要容器、虚拟机或更严格的权限隔离。

### 问题 4：模型选错工具怎么办？

工具错误不会直接让 Agent 崩溃，而是转换成结构化错误结果回传给模型。例如文件不存在、路径越界或测试命令不合法，模型都能看到具体原因并重新选择参数。

### 问题 5：为什么要同时支持 JSON 动作和原生 tool calls？

原生 tool calling 结构更规范，但不同兼容网关支持程度不同。双协议可以让项目在网关不支持 `tools` 参数时降级运行，提高兼容性。

### 问题 6：为什么工具结果使用 user 消息回填，而不是原生 tool role？

部分兼容网关不支持原生 tool role。统一使用 user 消息回填，可以让两种协议共用一套历史结构，降低 Provider 和上下文管理的复杂度。

### 问题 7：上下文压缩会不会造成信息丢失？

会存在一定风险，所以压缩是分层的。最近工具结果保留，大结果先落盘，旧结果只压缩详细内容，关键状态由结构化摘要保留。完整历史仍然在会话文件中，不会被真正删除。

### 问题 8：为什么不让模型直接生成完整修改后的文件？

完整重写可能覆盖用户没有要求修改的内容。精确替换和 diff 修改可以限定变更范围，并且在上下文不匹配时拒绝写入，安全性和可审计性更好。

### 问题 9：Agent 如何判断任务完成？

正常情况下由模型返回 `final`。此外还有最大步骤数和连续解析失败两个兜底条件。系统提示要求测试通过后立即返回 final，避免重复读取文件和重复确认。

### 问题 10：为什么要设置最大步骤数？

模型可能因为解析错误、工具错误或目标不明确陷入循环。最大步骤数可以限制时间和 Token 消耗，并让系统最终给出明确的未完成结果。

### 问题 11：这是不是一个多 Agent 系统？

不是。当前实现是“单 Agent + 多工具”，而不是多个 Agent 互相协作。当前任务围绕同一个代码工作区连续完成查看、修改和测试，单一上下文更容易保证状态一致，也避免多个 Agent 同时修改文件产生冲突。

### 问题 12：你的项目是不是只是 Prompt 工程？

不是。Prompt 只负责告诉模型任务规则，真正的 Agent 能力来自主循环、工具注册与执行、路径安全校验、输出协议解析、上下文压缩、会话持久化、重试和终止机制。这些模块都可以脱离具体模型单独测试。

### 问题 13：如何证明你的设计有效？

项目测试覆盖协议解析、工具安全、diff 修改、上下文压缩、会话恢复、流式输出、API 重试和 Agent 闭环。目前测试套件 91 项全部通过，离线演示也能完成“测试失败—修改代码—测试通过”的过程。

### 问题 14：你的设计还有哪些局限？

目前工具调用是串行的，不能充分利用只读工具的并行性；会话管理只有恢复功能，还没有完善的会话列表和删除功能；模型返回 final 前是否测试成功也主要依赖提示约束，而不是硬性状态机。

后续可以增加只读工具并行、强制验证状态机、更完善的会话管理，以及容器级执行隔离。

## 十二、答辩表达原则

遇到具体设计问题时，建议按照下面的顺序回答：

```text
先说明运行机制
    ↓
说明这个设计解决了什么问题
    ↓
解释为什么采用当前方案
    ↓
主动说明代价和边界
```

不要只说“系统可以做到什么”，还要说明：

- 哪一部分由模型完成；
- 哪一部分由本地代码完成；
- 为什么不能直接采用更简单的方案；
- 当前方案在哪些情况下仍然不够完善。
