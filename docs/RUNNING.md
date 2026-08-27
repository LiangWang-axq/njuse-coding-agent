# 运行说明

## 环境

Python 3.10+。项目运行时只使用 Python 标准库，不需要安装任何 Agent 框架。

## 配置

在项目根目录复制 `.env.example` 为 `.env`，填写模型服务配置：

```text
AGENT_BASE_URL=https://api.deepseek.com/v1
AGENT_API_KEY=你的本地密钥
AGENT_MODEL=deepseek-chat
AGENT_TIMEOUT_SECONDS=90
AGENT_MAX_STEPS=12
AGENT_RETRIES=2
AGENT_CONTEXT_CHARS=16000
```

也可以在 PowerShell 中设置：

```powershell
$env:AGENT_API_KEY = "你的本地密钥"
```

`.env` 已被工具层禁止读写，真实密钥不应提交到 Git、README 或视频中。

## 启动

当前工作区目录就是 Agent 唯一允许访问的工作区。有两种启动方式：

### 交互模式（推荐）

```powershell
python -m coding_agent
```

进入对话界面后直接输入任务即可：

```text
Coding Agent 交互模式
模型: deepseek-chat · 工作区: D:\...\demo\tasks\fix_me
直接输入任务开始对话，输入 /help 查看命令。
你 > 检查订单计算模块，修复 bug 并运行测试
Agent > 我先查看一下相关文件……
  [工具] read_file("orders.py") -> OK
  [工具] run_tests("python -m unittest ...") -> 通过
（本轮 5 步 · 累计 5 步）
你 >
```

可用命令：`/help`、`/status`（工作区/模型/累计步骤）、`/new`（清空历史开始新任务）、`/exit`（或 `/quit`）。`Ctrl+C` 在输入时退出程序，在任务执行中取消当前轮并保留历史；`Ctrl+D/Z`（EOF）正常退出。多轮任务在同一进程内共享历史。

### 一次性模式

```powershell
python -m coding_agent "检查 demo/tasks/fix_me，修复订单计算 bug，补充测试并运行测试"
```

执行完自动退出，退出码 0 表示成功、1 表示 Agent 错误、2 表示达到最大步骤未完成。模型回答同样流式输出，工具调用以 `[步骤 N]` 彩色行展示。

模型每轮返回一个 JSON 动作或原生 tool call。工具调用格式为：

```json
{"type":"tool_call","tool":"read_file","arguments":{"path":"README.txt"}}
```

完成格式为：

```json
{"type":"final","message":"已完成修改并通过测试"}
```

CLI 会打印每一步 `[步骤 N] 工具名(参数) -> OK/错误`，便于观察与录屏。

## 工具

`list_files`、`read_file`、`search_code`、`write_file`、`replace_in_file`、`run_tests`、`delete_file`、`move_file`、`git_status`。所有路径必须是工作区相对路径；`run_tests` 仅允许 `python -m unittest/pytest/compileall`，无 shell 拼接。任务针对工作区子目录时，`run_tests` 可用 `cwd` 参数指定相对目录（例如 `{"command": "python -m unittest discover -s tests", "cwd": "demo/tasks/fix_me"}`），测试会切换到该目录执行，避免导入路径错位。

## 上下文与重试

历史超出 `AGENT_CONTEXT_CHARS` 时自动丢弃最旧工具结果并插入压缩摘要；模型服务 429/5xx/超时按 `AGENT_RETRIES` 次退避重试（2s/4s）；网关不支持原生 tools 时自动降级为纯 JSON 协议。

## 验证

```powershell
python -m unittest discover -s tests -v
python demo/run_offline_demo.py
```
