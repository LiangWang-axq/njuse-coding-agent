软件工程专业推免项目：构建编程智能体

仓库地址：待创建公开仓库后填写

这是一个不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等框架的 Python 命令行 Coding Agent。它通过 OpenAI 兼容的 Chat Completions 接口与模型交互，自行维护对话历史，解析模型 JSON 动作，在当前工作区内读取文件、搜索代码、创建或修改文件，并运行受控测试命令。

运行：
1. 安装 Python 3.10 或更高版本。
2. 复制 .env.example 为 .env，填写 AGENT_API_KEY 和模型名；也可直接设置环境变量。
3. 在本目录执行：python -m coding_agent "检查项目并修复 bug，补充测试后运行测试"
4. 不传任务文本时会进入交互输入。

安全边界：所有文件操作只能使用工作区相对路径；拒绝绝对路径和路径穿越；禁止 Agent 写入 .env；测试命令不经过 shell，仅允许 python/pytest 的 unittest、pytest、compileall 入口，并设置超时。

测试：python -m unittest discover -s tests -v
离线演示：python demo/run_offline_demo.py

完整运行说明见 docs/RUNNING.md，真实任务步骤见 docs/DEMO.md。离线演示使用脚本化模型响应验证 Agent 的本地闭环，不需要暴露 API key；接入真实模型时使用上面的命令即可。
