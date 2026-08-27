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

在当前工作区目录执行，当前目录就是 Agent 唯一允许访问的工作区：

```powershell
python -m coding_agent "检查 demo/tasks/fix_me，修复订单计算 bug，补充测试并运行测试"
```

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

`list_files`、`read_file`、`search_code`、`write_file`、`replace_in_file`、`run_tests`、`delete_file`、`move_file`、`git_status`。所有路径必须是工作区相对路径；`run_tests` 仅允许 `python -m unittest/pytest/compileall`，无 shell 拼接。

## 上下文与重试

历史超出 `AGENT_CONTEXT_CHARS` 时自动丢弃最旧工具结果并插入压缩摘要；模型服务 429/5xx/超时按 `AGENT_RETRIES` 次退避重试（2s/4s）；网关不支持原生 tools 时自动降级为纯 JSON 协议。

## 验证

```powershell
python -m unittest discover -s tests -v
python demo/run_offline_demo.py
```
