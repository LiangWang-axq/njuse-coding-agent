# 运行说明

## 环境

Python 3.10+。项目运行时只使用 Python 标准库，不需要安装 Agent 框架。

## 配置

在项目根目录复制 `.env.example` 为 `.env`，填写模型服务配置：

```text
AGENT_BASE_URL=https://api.openai.com/v1
AGENT_API_KEY=你的本地密钥
AGENT_MODEL=模型名称
```

也可以在 PowerShell 中设置：

```powershell
$env:AGENT_API_KEY = "你的本地密钥"
$env:AGENT_MODEL = "模型名称"
```

`.env` 已被工具层禁止写入，真实密钥不应提交到 Git、README 或视频中。

## 启动

必须在项目根目录执行，当前目录就是 Agent 唯一允许访问的工作区：

```powershell
python -m coding_agent "修复 Todo 项目中的 bug，补充回归测试并运行测试"
```

模型每轮只能返回一个 JSON 对象。工具调用格式为：

```json
{"type":"tool_call","tool":"read_file","arguments":{"path":"README.txt"}}
```

完成格式为：

```json
{"type":"final","message":"已完成修改并通过测试"}
```

## 验证

```powershell
python -m unittest discover -s tests -v
python demo/run_offline_demo.py
```
