软件工程专业推免项目：构建编程智能体

仓库地址：https://github.com/LiangWang-axq/njuse-coding-agent

这是一个不依赖任何 Agent 框架（LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等）的 Python 命令行编程智能体。它通过 OpenAI 兼容接口调用大语言模型，自行维护对话历史与上下文压缩，解析模型返回的动作 JSON 或原生 tool calls，并在当前工作区内安全地读写、搜索、修改、删除、移动文件，查看 git 状态，运行受控测试命令，循环直至任务完成。

运行方法：
1. 安装 Python 3.10 及以上；
2. 复制 .env.example 为 .env，填写 AGENT_API_KEY（或设置环境变量）；
3. 交互对话（推荐）：在本目录执行 python -m coding_agent，直接输入任务开始对话；
4. 一次性执行：python -m coding_agent "任务描述"（执行完退出，退出码 0/1/2）；
5. 恢复历史会话：python -m coding_agent --resume（自动加载工作区内最近一次会话，交互/一次性模式均可用）。

交互命令：/help 查看帮助、/status 查看工作区/模型/会话/压缩统计/token 用量与累计步骤、/new 清空历史开始新会话、/exit（或 /quit）退出；Ctrl+C 在输入时退出、在任务执行中取消当前轮。

特色功能：
- 终端对话：模型回答逐字流式输出，工具调用以彩色多行卡片展示（工具名、参数、结果摘要、测试输出尾部），压缩发生时实时提示（如 `[上下文] L2 压缩 3 条旧工具结果`），每轮结束显示 token 用量/估算与预算（纯标准库 ANSI，管道输出自动去色）；
- 多轮会话：同一进程内持续对话；每条消息原子落盘到 <工作区>/.coding_agent/sessions/，重启后可用 --resume 恢复；
- 全本地工具执行：10 个工具全部自行实现，路径锁定在工作区内，拒绝越界、绝对路径与 .env；
- 四层上下文压缩：超大工具结果落盘（视图换预览）、超限裁切中间轮次、旧结果压成一行摘要、仍超限才调用模型生成结构化摘要；并用真实 usage 校准 token 估算，控制模型输入窗口；
- 输出解析：兼容 JSON 动作与原生 tool calling（含流式增量合并），支持一次多个工具调用；
- 容错与终止：429/5xx 与超时自动重试（2 秒/4 秒退避），工具错误回传模型修正，连续解析失败或达到最大步骤数时安全终止；系统提示内置收敛规则（测试通过立即返回 final、不重复读取已确认文件、验证语法用 compileall），防止模型陷入重复确认的无效循环；
- 测试命令白名单：仅允许 python -m unittest/pytest/compileall，禁止 shell 拼接。

验证：
python -m unittest discover -s tests -v
python demo/run_offline_demo.py

演示任务：demo/tasks/fix_me（含 2 个故意留下的 bug，供 agent 真实修复）。
