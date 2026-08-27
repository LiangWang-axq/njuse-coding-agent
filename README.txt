软件工程专业推免项目：构建编程智能体

仓库地址：https://github.com/LiangWang-axq/njuse-coding-agent

这是一个不依赖任何 Agent 框架（LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等）的 Python 命令行编程智能体。它通过 OpenAI 兼容接口调用大语言模型，自行维护对话历史与上下文压缩，解析模型返回的动作 JSON 或原生 tool calls，并在当前工作区内安全地读写、搜索、修改、删除、移动文件，查看 git 状态，运行受控测试命令，循环直至任务完成。

运行方法：
1. 安装 Python 3.10 及以上；
2. 复制 .env.example 为 .env，填写 AGENT_API_KEY（或设置环境变量）；
3. 在本目录执行：python -m coding_agent "任务描述"。

特色功能：
- 全本地工具执行：9 个工具全部自行实现，路径锁定在工作区内，拒绝越界、绝对路径与 .env；
- 自研上下文管理：超长历史自动压缩为摘要，控制模型输入窗口；
- 自研输出解析：兼容 JSON 动作与原生 tool calling，支持一次多个工具调用；
- 容错与终止：429/5xx 与超时自动重试（2 秒/4 秒退避），工具错误回传模型修正，连续解析失败或达到最大步骤数时安全终止；
- 测试命令白名单：仅允许 python -m unittest/pytest/compileall，禁止 shell 拼接。

验证：
python -m unittest discover -s tests -v
python demo/run_offline_demo.py

演示任务：demo/tasks/fix_me（含 2 个故意留下的 bug，供 agent 真实修复）。
