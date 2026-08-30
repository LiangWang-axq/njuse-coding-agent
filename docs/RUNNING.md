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
AGENT_MAX_STEPS=24
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

想接着上次的会话继续聊（跨进程/跨重启），加 `--resume`：

```powershell
python -m coding_agent --resume
```

也可以先列出会话，再按最新优先序号、完整 ID 或唯一 ID 前缀恢复指定会话：

```powershell
python -m coding_agent --list-sessions
python -m coding_agent --resume-session 2
python -m coding_agent --resume-session 20260829-161011
```

进入对话界面后直接输入任务即可：

```text
Coding Agent 交互模式
模型: deepseek-chat · 工作区: D:\...\demo\tasks\fix_me
直接输入任务开始对话，输入 /help 查看命令。
你 > 检查订单计算模块，修复 bug 并运行测试
Agent > 我先查看一下相关文件……
  [步骤 2] read_file
    参数: {
      "path": "orders.py"
    }
    结果: 成功 · orders.py · 812 字符
  [步骤 4] run_tests
    参数: {
      "command": "python -m unittest discover -s tests -v"
    }
    结果: 失败 (returncode 1)
        输出尾: FAILED (failures=2)
  [上下文] L2 压缩 3 条旧工具结果
（本轮 6 步 · 累计 6 步 · token prompt 4321 / completion 258 · 预算 4000）
你 >
```

可用命令：`/help`、`/status`（工作区/模型/会话/压缩统计/token 用量/累计步骤）、`/sessions`（最新优先列出历史）、`/resume <选择器>`（切换会话）、`/delete <选择器>`（确认后永久删除）、`/new`（开始新会话）、`/exit`（或 `/quit`）。选择器可以是列表序号、完整 ID 或唯一 ID 前缀。删除当前会话后会自动开始新会话；删除其他会话不影响当前上下文。

`Ctrl+C` 在输入时退出程序，在任务执行中取消当前轮并保留历史；`Ctrl+D/Z`（EOF）正常退出。多轮任务在同一进程内共享历史，且每条消息原子落盘到 `<工作区>/.coding_agent/sessions/`。

### 会话管理

列表和删除是纯本地操作，即使尚未配置 API key 也可以使用：

```powershell
python -m coding_agent --list-sessions
python -m coding_agent --delete-session 3
python -m coding_agent --delete-session 20260829-155609 --yes
```

删除默认展示会话摘要并询问 `[y/N]`；只有 `y` 或 `yes` 才会永久删除，`--yes` 用于明确跳过确认。损坏的 JSONL 会在列表中标记为“损坏”，不能恢复但仍可删除。删除只影响选中的 `.coding_agent/sessions/*.jsonl`，不会清理 `.coding_agent/results/` 中的上下文大结果文件。

### 一次性模式

```powershell
python -m coding_agent "检查 demo/tasks/fix_me，修复订单计算 bug，补充测试并运行测试"
python -m coding_agent --resume "继续上次任务"
```

执行完自动退出，退出码 0 表示成功、1 表示 Agent 错误或会话管理失败/取消、2 表示达到最大步骤未完成。模型回答同样流式输出，工具调用以 `[步骤 N]` 彩色行展示。

模型每轮返回一个 JSON 动作或原生 tool call。工具调用以多行卡片展示（工具名、参数、结果摘要、测试输出尾部）；压缩发生时打印 `[上下文] …` 提示，每轮结束显示 token 用量/估算与预算。工具动作格式为：

```json
{"type":"tool_call","tool":"read_file","arguments":{"path":"README.txt"}}
```

完成格式为：

```json
{"type":"final","message":"已完成修改并通过测试"}
```

CLI 会打印每一步 `[步骤 N] 工具名(参数) -> OK/错误`，便于观察与录屏。

## 工具

`list_files`、`read_file`、`search_code`、`write_file`、`replace_in_file`、`apply_diff`、`run_tests`、`delete_file`、`move_file`、`git_status`。所有路径必须是工作区相对路径；`run_tests` 仅允许 `python -m unittest/pytest/compileall`，无 shell 拼接。任务针对工作区子目录时，`run_tests` 可用 `cwd` 参数指定相对目录（例如 `{"command": "python -m unittest discover -s tests", "cwd": "demo/tasks/fix_me"}`），测试会切换到该目录执行，避免导入路径错位。`apply_diff` 接受标准 unified diff（`--- a/文件`、`+++ b/文件`、`@@ -起始,行数 +起始,行数 @@`，空格/`-`/`+` 分别表示上下文、删除、新增），hunk 头缺行号时按唯一匹配定位；一次可含多个 hunk，全部匹配才写入（原子应用），只修改已有文件。

## 上下文与重试

`ContextManager` 保存完整历史，调用模型前由 `prepare()` 生成压缩视图：超大工具结果先落盘到 `.coding_agent/results/`（视图保留预览），消息数/字符超限时依次做中间裁切、旧结果一行摘要，仍超限才调用模型生成 `<summary>` 结构化摘要；并用真实 `usage` 校准 token 估算。模型服务 429/5xx/超时按 `AGENT_RETRIES` 次退避重试（2s/4s）；网关不支持原生 tools 时自动降级为纯 JSON 协议。

## 验证

```powershell
python -m unittest discover -s tests -v
python demo/run_offline_demo.py
```
